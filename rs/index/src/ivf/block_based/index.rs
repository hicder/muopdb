use std::collections::{BinaryHeap, HashSet};
use std::sync::{Arc, RwLock};

use anyhow::Result;
use compression::compression::{AsyncIntSeqDecoder, AsyncIntSeqIterator};
use log::info;
use quantization::quantization::Quantizer;
use quantization::typing::VectorOps;
use utils::block_cache::BlockCache;
use utils::distance::l2::L2DistanceCalculator;
use utils::DistanceCalculator;

use crate::ivf::block_based::storage::BlockBasedPostingListStorage;
use crate::query::iters::InvertedIndexIter;
use crate::query::planner::Planner;
use crate::utils::{
    IdWithScore, IntermediateResult, PointAndDistance, SearchContext, SearchResult, SearchStats,
};
use crate::vector::async_storage::AsyncFixedFileVectorStorage;
use crate::vector::StorageContext;

pub struct BlockBasedIvf<Q: Quantizer>
where
    Q::QuantizedT: Send + Sync,
{
    vector_storage: AsyncFixedFileVectorStorage<Q::QuantizedT>,
    posting_list_storage: BlockBasedPostingListStorage,
    num_clusters: usize,
    quantizer: Q,
    invalid_point_ids: RwLock<HashSet<u32>>,
}

/// A block-based Inverted File Index (IVF) implementation.
///
/// This index uses asynchronous I/O and block caching for efficient search
/// on large-scale vector datasets that do not fit in memory.
impl<Q: Quantizer> BlockBasedIvf<Q>
where
    Q::QuantizedT: Send + Sync,
{
    /// Creates a new `BlockBasedIvf` index by loading components from the specified directory.
    ///
    /// # Arguments
    /// * `block_cache` - The shared block cache for file I/O.
    /// * `base_directory` - The directory containing the IVF index files.
    ///
    /// # Returns
    /// * `Result<Self>` - A new IVF index instance or an error if loading fails.
    pub async fn new(block_cache: Arc<BlockCache>, base_directory: String) -> Result<Self> {
        let index_storage = BlockBasedPostingListStorage::new(
            block_cache.clone(),
            format!("{}/index", base_directory),
        )
        .await?;

        let vector_storage = AsyncFixedFileVectorStorage::<Q::QuantizedT>::new(
            block_cache.clone(),
            format!("{}/vectors", base_directory),
            index_storage.header().quantized_dimension as usize,
        )
        .await?;

        let quantizer_directory = format!("{}/quantizer", base_directory);
        let quantizer = Q::read(quantizer_directory)?;

        let num_clusters = index_storage.header().num_clusters as usize;

        Ok(Self {
            vector_storage,
            posting_list_storage: index_storage,
            num_clusters,
            quantizer,
            invalid_point_ids: RwLock::new(HashSet::new()),
        })
    }

    /// Creates a new `BlockBasedIvf` index with specific file offsets for the index and vectors.
    ///
    /// # Arguments
    /// * `block_cache` - The shared block cache for file I/O.
    /// * `base_directory` - The directory containing the IVF index files.
    /// * `index_offset` - The byte offset within the index file where the IVF data starts.
    /// * `vector_offset` - The byte offset within the vector file where the vector data starts.
    ///
    /// # Returns
    /// * `Result<Self>` - A new IVF index instance or an error if loading fails.
    pub async fn new_with_offset(
        block_cache: Arc<BlockCache>,
        base_directory: String,
        index_offset: usize,
        vector_offset: usize,
    ) -> Result<Self> {
        let index_storage = BlockBasedPostingListStorage::new_with_offset(
            block_cache.clone(),
            format!("{}/index", base_directory),
            index_offset,
        )
        .await?;

        let vector_storage = AsyncFixedFileVectorStorage::<Q::QuantizedT>::new_with_offset(
            block_cache.clone(),
            format!("{}/vectors", base_directory),
            index_storage.header().quantized_dimension as usize,
            vector_offset,
        )
        .await?;

        let quantizer_directory = format!("{}/quantizer", base_directory);
        let quantizer = Q::read(quantizer_directory)?;

        let num_clusters = index_storage.header().num_clusters as usize;

        Ok(Self {
            vector_storage,
            posting_list_storage: index_storage,
            num_clusters,
            quantizer,
            invalid_point_ids: RwLock::new(HashSet::new()),
        })
    }

    /// Finds the centroids nearest to the given query vector.
    ///
    /// # Arguments
    /// * `vector` - The query vector to search for.
    /// * `num_probes` - The number of nearest centroids to return.
    ///
    /// # Returns
    /// * `Result<Vec<usize>>` - A list of indices for the nearest centroids.
    pub async fn find_nearest_centroids(
        &self,
        vector: &[f32],
        num_probes: usize,
    ) -> Result<Vec<usize>> {
        let mut distances: Vec<(usize, f32)> = Vec::new();
        for i in 0..self.num_clusters {
            let centroid = self.posting_list_storage.get_centroid(i).await?;
            let dist = L2DistanceCalculator::calculate(vector, &centroid);
            distances.push((i, dist));
        }
        distances.select_nth_unstable_by(num_probes - 1, |a, b| a.1.total_cmp(&b.1));
        let mut nearest_centroids: Vec<(usize, f32)> =
            distances.into_iter().take(num_probes).collect();
        nearest_centroids.sort_by(|a, b| a.1.total_cmp(&b.1));
        Ok(nearest_centroids.into_iter().map(|(idx, _)| idx).collect())
    }

    /// Scans a single posting list (cell) for vectors similar to the query.
    ///
    /// # Arguments
    /// * `centroid` - The index of the centroid/cell to scan.
    /// * `query` - The query vector.
    /// * `record_pages` - Whether to track block cache hits/misses.
    /// * `planner` - An optional search planner for additional filtering.
    ///
    /// # Returns
    /// * `Result<IntermediateResult>` - The search results from this posting list.
    async fn scan_posting_list(
        &self,
        centroid: usize,
        query: &[f32],
        record_pages: bool,
        planner: Option<Arc<Planner>>,
    ) -> Result<IntermediateResult> {
        info!(
            "[BLOCK-BASED] Scanning posting list for centroid {}",
            centroid
        );
        let mut context = SearchContext::new(record_pages);
        let decoder = self
            .posting_list_storage
            .get_posting_list_decoder(centroid)
            .await?;
        let mut iterator = decoder.into_iterator();

        let quantized_query = Q::QuantizedT::process_vector(query, &self.quantizer);
        let mut point_and_distances = Vec::new();

        while let Some(point_id_u64) = iterator.next().await? {
            let point_id = point_id_u64 as u32;
            let is_invalidated = {
                let invalidated_ids = self.invalid_point_ids.read().unwrap();
                invalidated_ids.contains(&point_id)
            };
            if is_invalidated {
                continue;
            }

            let vector = self.vector_storage.get(point_id, &mut context).await?;
            let distance = self.quantizer.distance(
                &quantized_query,
                &vector,
                utils::distance::l2::L2DistanceCalculatorImpl::StreamingSIMD,
            );
            point_and_distances.push(PointAndDistance::new(distance, point_id));
        }

        // sort by ids first for planner
        point_and_distances.sort_by_key(|pd| pd.point_id);

        if let Some(planner) = planner {
            let all_ids = point_and_distances
                .iter()
                .map(|pd| pd.point_id)
                .collect::<Vec<u32>>();
            let mut filtered_point_ids = HashSet::new();
            if let Ok(mut iter) = planner.plan_with_ids(&all_ids) {
                while let Some(id) = iter.next() {
                    filtered_point_ids.insert(id);
                }
            }
            point_and_distances.retain(|pd| filtered_point_ids.contains(&pd.point_id));
        }

        point_and_distances.sort_by(|a, b| a.distance.total_cmp(&b.distance));

        let mut stats = SearchStats::new();
        stats.num_pages_accessed = context.num_pages_accessed();

        Ok(IntermediateResult {
            point_and_distances,
            stats,
        })
    }

    /// Performs a search across multiple specified clusters/centroids.
    ///
    /// # Arguments
    /// * `query` - The query vector.
    /// * `nearest_centroid_ids` - The list of centroid indices to explore.
    /// * `k` - The number of nearest neighbors to return.
    /// * `record_pages` - Whether to track block cache hits/misses.
    /// * `planner` - An optional search planner for additional filtering.
    ///
    /// # Returns
    /// * `Result<IntermediateResult>` - The combined search results from all probed clusters.
    async fn search_with_centroids(
        &self,
        query: &[f32],
        nearest_centroid_ids: Vec<usize>,
        k: usize,
        record_pages: bool,
        planner: Option<Arc<Planner>>,
    ) -> Result<IntermediateResult> {
        let mut heap = BinaryHeap::with_capacity(k);
        let mut final_stats = SearchStats::new();

        for centroid in nearest_centroid_ids {
            let results = self
                .scan_posting_list(centroid, query, record_pages, planner.clone())
                .await?;
            for pd in results.point_and_distances {
                if heap.len() < k {
                    heap.push(pd);
                } else if let Some(max) = heap.peek() {
                    if pd < *max {
                        heap.pop();
                        heap.push(pd);
                    }
                }
            }
            final_stats.merge(&results.stats);
        }

        let mut results: Vec<PointAndDistance> = heap.into_vec();
        results.sort();

        Ok(IntermediateResult {
            point_and_distances: results,
            stats: final_stats,
        })
    }

    /// Performs a search across clusters and maps local point IDs back to document IDs.
    ///
    /// # Arguments
    /// * `query` - The query vector.
    /// * `nearest_centroid_ids` - The list of centroid indices to explore.
    /// * `k` - The number of nearest neighbors to return.
    /// * `record_pages` - Whether to track block cache hits/misses.
    /// * `planner` - An optional search planner for additional filtering.
    ///
    /// # Returns
    /// * `Result<SearchResult>` - The final search results containing document IDs and scores.
    pub async fn search_with_centroids_and_remap(
        &self,
        query: &[f32],
        nearest_centroid_ids: Vec<usize>,
        k: usize,
        record_pages: bool,
        planner: Option<Arc<Planner>>,
    ) -> Result<SearchResult> {
        let results = self
            .search_with_centroids(query, nearest_centroid_ids, k, record_pages, planner)
            .await?;

        let mut id_with_scores = Vec::with_capacity(results.point_and_distances.len());
        for pd in results.point_and_distances {
            let doc_id = self
                .posting_list_storage
                .get_doc_id(pd.point_id as usize)
                .await?;
            id_with_scores.push(IdWithScore {
                doc_id,
                score: *pd.distance,
            });
        }

        Ok(SearchResult {
            id_with_scores,
            stats: results.stats,
        })
    }

    /// Returns the total number of clusters in the index.
    ///
    /// # Returns
    /// * `usize` - The cluster count.
    pub fn num_clusters(&self) -> usize {
        self.num_clusters
    }

    /// Returns the total number of vectors stored in the index.
    ///
    /// # Returns
    /// * `usize` - The vector count.
    pub fn num_vectors(&self) -> usize {
        self.vector_storage.num_vectors()
    }

    pub async fn get_doc_id(&self, point_id: u32) -> Result<u128> {
        self.posting_list_storage
            .get_doc_id(point_id as usize)
            .await
    }

    /// Retrieves the vector data for a given point ID.
    ///
    /// # Arguments
    /// * `point_id` - The internal point ID.
    /// * `context` - A storage context for tracking cache stats.
    ///
    /// # Returns
    /// * `Result<Vec<Q::QuantizedT>>` - The vector data if found, or an error.
    pub async fn get_vector(
        &self,
        point_id: u32,
        context: &mut impl StorageContext,
    ) -> Result<Vec<Q::QuantizedT>> {
        self.vector_storage.get(point_id, context).await
    }

    /// Performs a complete IVF search for the given query.
    ///
    /// # Arguments
    /// * `query` - The query vector.
    /// * `k` - The number of nearest neighbors to return.
    /// * `num_probes` - The number of clusters to probe.
    /// * `record_pages` - Whether to track block cache hits/misses.
    /// * `planner` - An optional search planner for additional filtering.
    ///
    /// # Returns
    /// * `Result<Option<SearchResult>>` - The search results, or `None` if no results were found.
    pub async fn search(
        &self,
        query: &[f32],
        k: usize,
        num_probes: u32,
        record_pages: bool,
        planner: Option<Arc<Planner>>,
    ) -> Result<Option<SearchResult>> {
        let nearest_centroids = self
            .find_nearest_centroids(query, num_probes as usize)
            .await?;
        let results = self
            .search_with_centroids_and_remap(query, nearest_centroids, k, record_pages, planner)
            .await?;

        Ok(Some(results))
    }

    /// Marks a document ID as invalid/deleted in the index.
    ///
    /// # Arguments
    /// * `doc_id` - The document ID to invalidate.
    ///
    /// # Returns
    /// * `Result<bool>` - `true` if the document was successfully marked as invalid, `false` if it wasn't found.
    pub async fn invalidate(&self, doc_id: u128) -> Result<bool> {
        match self.get_point_id(doc_id).await? {
            Some(point_id) => Ok(self.invalid_point_ids.write().unwrap().insert(point_id)),
            None => Ok(false),
        }
    }

    /// Marks a batch of document IDs as invalid/deleted.
    ///
    /// # Arguments
    /// * `doc_ids` - The list of document IDs to invalidate.
    ///
    /// # Returns
    /// * `Result<Vec<u128>>` - The list of document IDs that were successfully invalidated.
    pub async fn invalidate_batch(&self, doc_ids: &[u128]) -> Result<Vec<u128>> {
        let mut successfully_invalidated = Vec::new();
        for &doc_id in doc_ids {
            if self.invalidate(doc_id).await? {
                successfully_invalidated.push(doc_id);
            }
        }
        Ok(successfully_invalidated)
    }

    /// Checks if a document ID has been marked as invalid.
    ///
    /// # Arguments
    /// * `doc_id` - The document ID to check.
    ///
    /// # Returns
    /// * `Result<bool>` - `true` if invalid, `false` otherwise.
    pub async fn is_invalidated(&self, doc_id: u128) -> Result<bool> {
        match self.get_point_id(doc_id).await? {
            Some(point_id) => Ok(self.invalid_point_ids.read().unwrap().contains(&point_id)),
            None => Ok(false),
        }
    }

    /// Resolves a document ID to its internal point ID.
    ///
    /// # Arguments
    /// * `doc_id` - The document ID to lookup.
    ///
    /// # Returns
    /// * `Result<Option<u32>>` - The internal point ID if found, or `None`.
    async fn get_point_id(&self, doc_id: u128) -> Result<Option<u32>> {
        for point_id in 0..self.vector_storage.num_vectors() {
            if let Ok(stored_doc_id) = self.posting_list_storage.get_doc_id(point_id).await {
                if stored_doc_id == doc_id {
                    return Ok(Some(point_id as u32));
                }
            }
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use compression::elias_fano::ef::EliasFano;
    use quantization::noq::noq::NoQuantizer;
    use quantization::quantization::WritableQuantizer;
    use tempdir::TempDir;
    use utils::block_cache::cache::BlockCacheConfig;
    use utils::distance::l2::L2DistanceCalculator;
    use utils::test_utils::generate_random_vector;

    use super::*;
    use crate::ivf::builder::{IvfBuilder, IvfBuilderConfig};
    use crate::ivf::writer::IvfWriter;

    #[tokio::test]
    async fn test_block_based_ivf_search() {
        let temp_dir = TempDir::new("test_block_based_ivf_search").unwrap();
        let base_directory = temp_dir.path().to_str().unwrap().to_string();

        let num_clusters = 5;
        let num_vectors = 100;
        let num_features = 4;

        let quantizer = NoQuantizer::<L2DistanceCalculator>::new(num_features);
        let quantizer_directory = format!("{}/quantizer", base_directory);
        fs::create_dir_all(&quantizer_directory).unwrap();
        quantizer.write_to_directory(&quantizer_directory).unwrap();

        let mut builder = IvfBuilder::new(IvfBuilderConfig {
            max_iteration: 100,
            batch_size: 4,
            num_clusters,
            num_data_points_for_clustering: num_vectors,
            max_clusters_per_vector: 1,
            distance_threshold: 0.1,
            base_directory: base_directory.clone(),
            memory_size: 1024,
            file_size: 4096,
            num_features,
            tolerance: 0.0,
            max_posting_list_size: usize::MAX,
        })
        .unwrap();

        for i in 0..num_vectors {
            builder
                .add_vector(i as u128, &generate_random_vector(num_features))
                .unwrap();
        }
        builder.build().unwrap();

        let writer =
            IvfWriter::<_, EliasFano, L2DistanceCalculator>::new(base_directory.clone(), quantizer);
        writer.write(&mut builder, false).unwrap();

        let config = BlockCacheConfig::default();
        let block_cache = Arc::new(BlockCache::new(config));

        let block_based_ivf =
            BlockBasedIvf::<NoQuantizer<L2DistanceCalculator>>::new(block_cache, base_directory)
                .await
                .unwrap();

        let query = generate_random_vector(num_features);
        let results = block_based_ivf
            .search(&query, 5, 2, false, None)
            .await
            .unwrap();

        assert!(results.is_some());
        let search_result = results.unwrap();
        assert_eq!(search_result.id_with_scores.len(), 5);

        // Verify scores are ascending
        for i in 1..search_result.id_with_scores.len() {
            assert!(
                search_result.id_with_scores[i - 1].score <= search_result.id_with_scores[i].score
            );
        }
    }

    #[tokio::test]
    async fn test_block_based_ivf_invalidation() {
        let temp_dir = TempDir::new("test_block_based_ivf_invalidation").unwrap();
        let base_directory = temp_dir.path().to_str().unwrap().to_string();

        let num_clusters = 2;
        let num_vectors = 10;
        let num_features = 4;

        let quantizer = NoQuantizer::<L2DistanceCalculator>::new(num_features);
        let quantizer_directory = format!("{}/quantizer", base_directory);
        fs::create_dir_all(&quantizer_directory).unwrap();
        quantizer.write_to_directory(&quantizer_directory).unwrap();

        let mut builder = IvfBuilder::new(IvfBuilderConfig {
            max_iteration: 10,
            batch_size: 2,
            num_clusters,
            num_data_points_for_clustering: num_vectors,
            max_clusters_per_vector: 1,
            distance_threshold: 0.1,
            base_directory: base_directory.clone(),
            memory_size: 1024,
            file_size: 4096,
            num_features,
            tolerance: 0.0,
            max_posting_list_size: usize::MAX,
        })
        .unwrap();

        for i in 0..num_vectors {
            builder
                .add_vector(i as u128, &generate_random_vector(num_features))
                .unwrap();
        }
        builder.build().unwrap();

        let writer =
            IvfWriter::<_, EliasFano, L2DistanceCalculator>::new(base_directory.clone(), quantizer);
        writer.write(&mut builder, false).unwrap();

        let config = BlockCacheConfig::default();
        let block_cache = Arc::new(BlockCache::new(config));

        let block_based_ivf =
            BlockBasedIvf::<NoQuantizer<L2DistanceCalculator>>::new(block_cache, base_directory)
                .await
                .unwrap();

        let query = generate_random_vector(num_features);
        let results_before = block_based_ivf
            .search(&query, 5, 2, false, None)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(results_before.id_with_scores.len(), 5);

        // Invalidate the first result
        let doc_id_to_invalidate = results_before.id_with_scores[0].doc_id;
        block_based_ivf
            .invalidate(doc_id_to_invalidate)
            .await
            .unwrap();

        let results_after = block_based_ivf
            .search(&query, 5, 2, false, None)
            .await
            .unwrap()
            .unwrap();

        for res in results_after.id_with_scores {
            assert_ne!(res.doc_id, doc_id_to_invalidate);
        }
    }
}
