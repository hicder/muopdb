use std::cmp::Ordering;
use std::sync::Arc;

use config::search_params::SearchParams;
use quantization::noq::noq::NoQuantizer;
use quantization::quantization::Quantizer;
use tracing::debug;
use utils::distance::l2::L2DistanceCalculator;

use crate::hnsw::block_based::index::BlockBasedHnsw;
use crate::ivf::block_based::index::BlockBasedIvf;
use crate::query::planner::Planner;
use crate::utils::SearchResult;

pub struct Spann<Q: Quantizer>
where
    Q::QuantizedT: Send + Sync,
{
    centroids: BlockBasedHnsw<NoQuantizer<L2DistanceCalculator>>,
    posting_lists: BlockBasedIvf<Q>,
}

impl<Q: Quantizer> Spann<Q>
where
    Q::QuantizedT: Send + Sync,
{
    /// Creates a new `Spann` index using block-based HNSW for centroids.
    ///
    /// # Arguments
    /// * `centroids` - The block-based HNSW index for centroids.
    /// * `posting_lists` - The IVF posting lists.
    ///
    /// # Returns
    /// * `Self` - A new `Spann` instance.
    pub fn new(
        centroids: BlockBasedHnsw<NoQuantizer<L2DistanceCalculator>>,
        posting_lists: BlockBasedIvf<Q>,
    ) -> Self {
        Self {
            centroids,
            posting_lists,
        }
    }

    /// Returns a reference to the centroids index.
    ///
    /// # Returns
    /// * `&BlockBasedHnsw` - A reference to the underlying centroids storage.
    #[cfg(test)]
    pub fn get_centroids(&self) -> &BlockBasedHnsw<NoQuantizer<L2DistanceCalculator>> {
        &self.centroids
    }

    /// Returns a reference to the posting lists.
    ///
    /// # Returns
    /// * `&BlockBasedIvf<Q>` - A reference to the underlying IVF storage.
    #[cfg(test)]
    pub fn get_posting_lists(&self) -> &BlockBasedIvf<Q> {
        &self.posting_lists
    }

    /// Returns the document ID associated with a point ID.
    ///
    /// # Arguments
    /// * `point_id` - The internal point ID.
    ///
    /// # Returns
    /// * `Option<u128>` - The 128-bit document ID if found, or `None`.
    pub async fn get_doc_id(&self, point_id: u32) -> Option<u128> {
        self.posting_lists.get_doc_id(point_id).await.ok()
    }

    /// Returns the point ID associated with a document ID.
    ///
    /// # Warning
    /// This is very expensive and should only be used for testing as it performs a scan.
    ///
    /// # Arguments
    /// * `doc_id` - The 128-bit document ID.
    ///
    /// # Returns
    /// * `Option<u32>` - The internal point ID if found, or `None`.
    #[cfg(test)]
    pub async fn get_point_id(&self, doc_id: u128) -> Option<u32> {
        self.posting_lists.get_point_id(doc_id).await.ok().flatten()
    }

    /// Retrieves the quantized vector data for a point ID.
    ///
    /// # Arguments
    /// * `point_id` - The internal point ID.
    ///
    /// # Returns
    /// * `Option<Vec<Q::QuantizedT>>` - The quantized vector data if found.
    pub async fn get_vector(&self, point_id: u32) -> Option<Vec<Q::QuantizedT>> {
        use crate::utils::SearchContext;
        let mut context = SearchContext::new(false);
        self.posting_lists
            .get_vector(point_id, &mut context)
            .await
            .ok()
    }

    /// Invalidates a document ID in the index.
    ///
    /// # Arguments
    /// * `doc_id` - The document ID to invalidate.
    ///
    /// # Returns
    /// * `bool` - `true` if the document was successfully invalidated, `false` otherwise.
    pub async fn invalidate(&self, doc_id: u128) -> bool {
        self.posting_lists.invalidate(doc_id).await.unwrap_or(false)
    }

    /// Invalidates a batch of document IDs.
    ///
    /// # Arguments
    /// * `doc_ids` - A slice of document IDs to invalidate.
    ///
    /// # Returns
    /// * `Vec<u128>` - The list of document IDs that were successfully invalidated.
    pub async fn invalidate_batch(&self, doc_ids: &[u128]) -> Vec<u128> {
        self.posting_lists
            .invalidate_batch(doc_ids)
            .await
            .unwrap_or_default()
    }

    /// Checks if a document ID is invalidated.
    ///
    /// # Arguments
    /// * `doc_id` - The document ID to check.
    ///
    /// # Returns
    /// * `bool` - `true` if the document is invalidated, `false` otherwise.
    pub async fn is_invalidated(&self, doc_id: u128) -> bool {
        self.posting_lists
            .is_invalidated(doc_id)
            .await
            .unwrap_or(false)
    }

    /// Returns the total number of documents in the index.
    ///
    /// # Returns
    /// * `usize` - The document count.
    pub fn count_of_all_documents(&self) -> usize {
        self.posting_lists.num_vectors()
    }

    /// Prints all posting lists for debugging purposes.
    ///
    /// Output format:
    /// * [centroid_id] -> (point_id, doc_id, [v1, v2, ...]), (point_id, doc_id, [v1, v2, ...]), ...
    ///
    /// # Warning
    /// This is very expensive and should only be used for debugging/testing.
    #[cfg(test)]
    pub async fn debug_print_posting_lists(&self) {
        use compression::compression::{AsyncIntSeqDecoder, AsyncIntSeqIterator};

        use crate::utils::SearchContext;

        let num_clusters = self.posting_lists.num_clusters();
        let mut context = SearchContext::new(false);
        for centroid_id in 0..num_clusters {
            let decoder = self
                .posting_lists
                .get_posting_list_decoder(centroid_id)
                .await
                .expect("Failed to get posting list decoder");
            let mut iterator = AsyncIntSeqDecoder::into_iterator(decoder);

            let mut entries = Vec::new();
            while let Some(point_id_u64) = iterator.next().await.expect("Failed to iterate") {
                let point_id = point_id_u64 as u32;
                let doc_id = self.get_doc_id(point_id).await.unwrap_or(0);
                let raw_vec = self.get_vector(point_id).await.unwrap_or_default();
                entries.push(format!("({}, {}, {:?})", point_id, doc_id, raw_vec));
            }

            let centroid_vec = self
                .centroids
                .get_vector(centroid_id as u32, &mut context)
                .await
                .unwrap_or_default();
            eprintln!(
                "* [{}] {:?} -> {}",
                centroid_id,
                centroid_vec,
                entries.join(", ")
            );
        }
    }
}

impl<Q: Quantizer> Spann<Q>
where
    Q::QuantizedT: Send + Sync,
{
    /// Performs a search for the nearest neighbors of a query vector.
    ///
    /// # Arguments
    /// * `query` - The query vector.
    /// * `params` - Search parameters including top_k and num_probes.
    /// * `planner` - An optional search planner for additional filtering.
    ///
    /// # Returns
    /// * `Option<SearchResult>` - The search results if any, or `None`.
    pub async fn search(
        &self,
        query: Vec<f32>,
        params: &SearchParams,
        planner: Option<Arc<Planner>>,
    ) -> Option<SearchResult> {
        let num_explored_centroids = params.num_explored_centroids();
        let nearest_centroids = self
            .centroids
            .ann_search(
                &query,
                num_explored_centroids,
                params.ef_construction,
                params.record_pages,
            )
            .await;
        let centroid_search_stats = nearest_centroids.stats;
        let nearest_centroid_ids = nearest_centroids.id_with_scores;
        if nearest_centroid_ids.is_empty() {
            return None;
        }

        let nearest_distance = nearest_centroid_ids
            .iter()
            .map(|pad| pad.score)
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Greater))
            .expect("nearest_distance should not be None");

        let nearest_centroid_ids: Vec<usize> = nearest_centroid_ids
            .iter()
            .filter(|centroid_and_distance| {
                centroid_and_distance.score - nearest_distance
                    <= nearest_distance * params.centroid_distance_ratio
            })
            .map(|x| x.doc_id as usize)
            .collect();

        debug!(
            "Number of nearest centroids: {}",
            nearest_centroid_ids.len()
        );

        let mut results = self
            .posting_lists
            .search_with_centroids_and_remap(
                &query,
                nearest_centroid_ids,
                params.top_k,
                params.record_pages,
                planner,
            )
            .await
            .ok()?;
        results.stats.merge(&centroid_search_stats);
        Some(results)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use config::enums::{IntSeqEncodingType, QuantizerType};
    use config::search_params::SearchParams;
    use quantization::noq::noq::NoQuantizer;
    use quantization::pq::pq::ProductQuantizer;
    use utils::distance::l2::L2DistanceCalculator;
    use utils::file_io::env::{DefaultEnv, Env, EnvConfig, FileType};

    use crate::spann::builder::{SpannBuilder, SpannBuilderConfig};
    use crate::spann::reader::SpannReader;
    use crate::spann::writer::SpannWriter;

    fn create_env() -> Arc<Box<dyn Env>> {
        let config = EnvConfig {
            file_type: FileType::CachedStandard,
            ..EnvConfig::default()
        };
        Arc::new(Box::new(DefaultEnv::new(config)))
    }

    #[tokio::test]
    async fn test_spann_search() {
        let temp_dir = tempdir::TempDir::new("spann_search_test")
            .expect("Failed to create temporary directory");
        let base_dir = temp_dir
            .path()
            .to_str()
            .expect("Failed to convert temporary directory path to string")
            .to_string();

        let num_clusters = 10;
        let num_vectors = 1000;
        let num_features = 4;
        let file_size = 4096;
        let balance_factor = 0.0;
        let max_posting_list_size = usize::MAX;
        let mut builder = SpannBuilder::new(SpannBuilderConfig {
            centroids_max_neighbors: 10,
            centroids_max_layers: 2,
            centroids_ef_construction: 100,
            centroids_vector_storage_memory_size: 1024,
            centroids_vector_storage_file_size: file_size,
            num_features,
            pq_subvector_dimension: 8,
            pq_num_bits: 8,
            pq_num_training_rows: 50,
            quantizer_type: QuantizerType::NoQuantizer,
            pq_max_iteration: 1000,
            pq_batch_size: 4,
            ivf_num_clusters: num_clusters,
            ivf_num_data_points_for_clustering: num_vectors,
            ivf_max_clusters_per_vector: 1,
            ivf_distance_threshold: 0.1,
            posting_list_encoding_type: IntSeqEncodingType::EliasFano,
            ivf_base_directory: base_dir.clone(),
            ivf_vector_storage_memory_size: 1024,
            ivf_vector_storage_file_size: file_size,
            centroids_clustering_tolerance: balance_factor,
            ivf_max_posting_list_size: max_posting_list_size,
            reindex: false,
        })
        .unwrap();

        // Generate 1000 vectors of f32, dimension 4
        for i in 0..num_vectors {
            builder
                .add(i as u128, &[i as f32, i as f32, i as f32, i as f32])
                .unwrap();
        }
        assert!(builder.build().is_ok());
        let spann_writer = SpannWriter::new(base_dir.clone());
        assert!(spann_writer.write(&mut builder).is_ok());

        let env = create_env();
        let spann_reader = SpannReader::new(base_dir.clone());
        let spann = spann_reader
            .read::<NoQuantizer<L2DistanceCalculator>>(env)
            .await
            .unwrap();

        let query = vec![2.4, 3.4, 4.4, 5.4];
        let k = 2;
        let num_probes = 2;

        let params = SearchParams::new(k, num_probes, false);

        let results = spann
            .search(query, &params, None)
            .await
            .expect("IVF search should return a result");

        assert_eq!(results.id_with_scores.len(), k);
        assert_eq!(results.id_with_scores[0].doc_id, 4); // Closest to [4.0, 4.0, 4.0, 4.0]
        assert_eq!(results.id_with_scores[1].doc_id, 3); // Next is [3.0, 3.0, 3.0, 3.0]
    }

    #[tokio::test]
    async fn test_spann_search_with_invalidation() {
        let temp_dir = tempdir::TempDir::new("spann_search_with_invalidation_test")
            .expect("Failed to create temporary directory");
        let base_dir = temp_dir
            .path()
            .to_str()
            .expect("Failed to convert temporary directory path to string")
            .to_string();

        let num_clusters = 10;
        let num_vectors = 1000;
        let num_features = 4;
        let file_size = 4096;
        let balance_factor = 0.0;
        let max_posting_list_size = usize::MAX;
        let mut builder = SpannBuilder::new(SpannBuilderConfig {
            centroids_max_neighbors: 10,
            centroids_max_layers: 2,
            centroids_ef_construction: 100,
            centroids_vector_storage_memory_size: 1024,
            centroids_vector_storage_file_size: file_size,
            num_features,
            pq_subvector_dimension: 8,
            pq_num_bits: 8,
            pq_num_training_rows: 50,
            quantizer_type: QuantizerType::NoQuantizer,
            pq_max_iteration: 1000,
            pq_batch_size: 4,
            ivf_num_clusters: num_clusters,
            ivf_num_data_points_for_clustering: num_vectors,
            ivf_max_clusters_per_vector: 1,
            ivf_distance_threshold: 0.1,
            posting_list_encoding_type: IntSeqEncodingType::EliasFano,
            ivf_base_directory: base_dir.clone(),
            ivf_vector_storage_memory_size: 1024,
            ivf_vector_storage_file_size: file_size,
            centroids_clustering_tolerance: balance_factor,
            ivf_max_posting_list_size: max_posting_list_size,
            reindex: false,
        })
        .unwrap();

        // Generate 1000 vectors of f32, dimension 4
        for i in 0..num_vectors {
            builder
                .add(i as u128, &[i as f32, i as f32, i as f32, i as f32])
                .unwrap();
        }
        assert!(builder.build().is_ok());
        let spann_writer = SpannWriter::new(base_dir.clone());
        assert!(spann_writer.write(&mut builder).is_ok());

        let env = create_env();
        let spann_reader = SpannReader::new(base_dir.clone());
        let spann = spann_reader
            .read::<NoQuantizer<L2DistanceCalculator>>(env)
            .await
            .unwrap();

        let query = vec![2.4, 3.4, 4.4, 5.4];
        let k = 2;
        let num_probes = 2;

        assert!(spann.invalidate(4).await);
        assert!(spann.is_invalidated(4).await);

        let params = SearchParams::new(k, num_probes, false);

        let results = spann
            .search(query, &params, None)
            .await
            .expect("IVF search should return a result");

        assert_eq!(results.id_with_scores.len(), k);
        assert_eq!(results.id_with_scores[0].doc_id, 3); // Closest to [3.0, 3.0, 3.0, 3.0]
        assert_eq!(results.id_with_scores[1].doc_id, 5); // Next is [5.0, 5.0, 5.0, 5.0]
    }

    #[tokio::test]
    async fn test_spann_search_with_pq() {
        let temp_dir = tempdir::TempDir::new("spann_search_with_pq_test")
            .expect("Failed to create temporary directory");
        let base_dir = temp_dir
            .path()
            .to_str()
            .expect("Failed to convert temporary directory path to string")
            .to_string();

        let num_clusters = 10;
        let num_vectors = 1000;
        let num_features = 4;
        let file_size = 4096;
        let balance_factor = 0.0;
        let max_posting_list_size = usize::MAX;
        let mut builder = SpannBuilder::new(SpannBuilderConfig {
            centroids_max_neighbors: 10,
            centroids_max_layers: 2,
            centroids_ef_construction: 100,
            centroids_vector_storage_memory_size: 1024,
            centroids_vector_storage_file_size: file_size,
            num_features,
            pq_subvector_dimension: 2,
            pq_num_bits: 2,
            pq_num_training_rows: 200,
            quantizer_type: QuantizerType::ProductQuantizer,
            pq_max_iteration: 1000,
            pq_batch_size: 4,
            ivf_num_clusters: num_clusters,
            ivf_num_data_points_for_clustering: num_vectors,
            ivf_max_clusters_per_vector: 1,
            ivf_distance_threshold: 0.1,
            posting_list_encoding_type: IntSeqEncodingType::EliasFano,
            ivf_base_directory: base_dir.clone(),
            ivf_vector_storage_memory_size: 1024,
            ivf_vector_storage_file_size: file_size,
            centroids_clustering_tolerance: balance_factor,
            ivf_max_posting_list_size: max_posting_list_size,
            reindex: false,
        })
        .unwrap();

        // Generate 1000 vectors of f32, dimension 4
        for i in 0..num_vectors {
            builder
                .add(i as u128, &[i as f32, i as f32, i as f32, i as f32])
                .unwrap();
        }
        assert!(builder.build().is_ok());
        let spann_writer = SpannWriter::new(base_dir.clone());
        assert!(spann_writer.write(&mut builder).is_ok());

        let env = create_env();
        let spann_reader = SpannReader::new(base_dir.clone());
        let spann = spann_reader
            .read::<ProductQuantizer<L2DistanceCalculator>>(env)
            .await
            .unwrap();

        let query = vec![2.4, 3.4, 4.4, 5.4];
        let k = 5;
        let num_probes = 2;

        let params = SearchParams::new(k, num_probes, false);

        let results = spann
            .search(query, &params, None)
            .await
            .expect("IVF search should return a result");

        assert_eq!(results.id_with_scores.len(), k);
        // subvector_dimension = 2 so very lossy!
        assert_eq!(results.id_with_scores[0].score, 0.0);
        assert_eq!(results.id_with_scores[1].score, 0.0);
        assert_eq!(results.id_with_scores[2].score, 0.0);
        assert_eq!(results.id_with_scores[3].score, 0.0);
        assert_eq!(results.id_with_scores[4].score, 0.0);
    }

    #[tokio::test]
    async fn test_spann_debug_print_posting_lists() {
        let temp_dir = tempdir::TempDir::new("spann_debug_print_test")
            .expect("Failed to create temporary directory");
        let base_dir = temp_dir
            .path()
            .to_str()
            .expect("Failed to convert temporary directory path to string")
            .to_string();

        let num_clusters = 3;
        let num_vectors = 20;
        let num_features = 4;
        let file_size = 4096;
        let balance_factor = 0.0;
        let max_posting_list_size = usize::MAX;
        let mut builder = SpannBuilder::new(SpannBuilderConfig {
            centroids_max_neighbors: 10,
            centroids_max_layers: 2,
            centroids_ef_construction: 100,
            centroids_vector_storage_memory_size: 1024,
            centroids_vector_storage_file_size: file_size,
            num_features,
            pq_subvector_dimension: 8,
            pq_num_bits: 8,
            pq_num_training_rows: 50,
            quantizer_type: QuantizerType::NoQuantizer,
            pq_max_iteration: 1000,
            pq_batch_size: 4,
            ivf_num_clusters: num_clusters,
            ivf_num_data_points_for_clustering: num_vectors,
            ivf_max_clusters_per_vector: 1,
            ivf_distance_threshold: 0.1,
            posting_list_encoding_type: IntSeqEncodingType::EliasFano,
            ivf_base_directory: base_dir.clone(),
            ivf_vector_storage_memory_size: 1024,
            ivf_vector_storage_file_size: file_size,
            centroids_clustering_tolerance: balance_factor,
            ivf_max_posting_list_size: max_posting_list_size,
            reindex: false,
        })
        .unwrap();

        // Generate 10 vectors of f32, dimension 4
        for i in 0..num_vectors {
            builder
                .add(i as u128, &[i as f32, i as f32, i as f32, i as f32])
                .unwrap();
        }
        assert!(builder.build().is_ok());
        let spann_writer = SpannWriter::new(base_dir.clone());
        assert!(spann_writer.write(&mut builder).is_ok());

        let env = create_env();
        let spann_reader = SpannReader::new(base_dir.clone());
        let spann = spann_reader
            .read::<NoQuantizer<L2DistanceCalculator>>(env)
            .await
            .unwrap();

        // This will print all posting lists to stderr
        // Format: * [centroid_id] -> (point_id, doc_id, [v1, v2, ...]), ...
        eprintln!("\n=== Debug Print Posting Lists ===");
        spann.debug_print_posting_lists().await;
        eprintln!("=== End Debug Print ===\n");

        // Verify the index has the expected number of documents
        assert_eq!(spann.count_of_all_documents(), num_vectors);
    }
}
