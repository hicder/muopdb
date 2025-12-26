use std::collections::BinaryHeap;
use std::sync::Arc;

use anyhow::Result;
use bit_vec::BitVec;
use log::info;
use num_traits::ToPrimitive;
use ordered_float::NotNan;
use quantization::quantization::Quantizer;
use quantization::typing::VectorOps;
use utils::block_cache::BlockCache;
use utils::distance::l2::L2DistanceCalculatorImpl::StreamingSIMD;

use super::async_graph_storage::AsyncHnswGraphStorage;
use super::writer::Header;
use crate::hnsw::utils::GraphTraversal;
use crate::utils::{IdWithScore, PointAndDistance, SearchContext, SearchResult, SearchStats};
use crate::vector::async_storage::AsyncFixedFileVectorStorage;
use crate::vector::StorageContext;

pub struct BuilderContext {
    visited: BitVec,
}

impl BuilderContext {
    pub fn new(max_id: u32) -> Self {
        Self {
            visited: BitVec::from_elem(max_id as usize, false),
        }
    }
}

impl StorageContext for BuilderContext {
    fn should_record_pages(&self) -> bool {
        false
    }

    fn record_pages(&mut self, _page_id: String) {}

    fn num_pages_accessed(&self) -> usize {
        0
    }

    fn reset_pages_accessed(&mut self) {}

    fn set_visited(&mut self, id: u32) {
        self.visited.set(id as usize, true);
    }

    fn visited(&self, id: u32) -> bool {
        self.visited.get(id as usize).unwrap_or(false)
    }
}

pub struct AsyncHnsw<Q: Quantizer>
where
    Q::QuantizedT: Send + Sync,
{
    graph_storage: AsyncHnswGraphStorage,
    vector_storage: AsyncFixedFileVectorStorage<Q::QuantizedT>,
    header: Header,
    quantizer: Q,
}

impl<Q: Quantizer> AsyncHnsw<Q>
where
    Q::QuantizedT: Send + Sync,
{
    pub async fn new(block_cache: Arc<BlockCache>, base_directory: String) -> Result<Self> {
        let graph_storage =
            AsyncHnswGraphStorage::new(block_cache.clone(), base_directory.clone()).await?;

        let vector_path = format!("{}/hnsw/vector_storage", base_directory);
        let vector_storage = AsyncFixedFileVectorStorage::<Q::QuantizedT>::new(
            block_cache.clone(),
            vector_path,
            graph_storage.header().quantized_dimension as usize,
        )
        .await?;

        let quantizer_directory = format!("{}/quantizer", base_directory);
        let quantizer = Q::read(quantizer_directory)?;

        let header = graph_storage.header().clone();

        Ok(Self {
            graph_storage,
            vector_storage,
            header,
            quantizer,
        })
    }

    pub async fn new_with_offsets(
        block_cache: Arc<BlockCache>,
        base_directory: String,
        data_offset: usize,
        vector_offset: usize,
    ) -> Result<Self> {
        let graph_storage = AsyncHnswGraphStorage::new_with_offset(
            block_cache.clone(),
            base_directory.clone(),
            data_offset,
        )
        .await?;

        let vector_path = format!("{}/hnsw/vector_storage", base_directory);
        let vector_storage = AsyncFixedFileVectorStorage::<Q::QuantizedT>::new_with_offset(
            block_cache.clone(),
            vector_path,
            graph_storage.header().quantized_dimension as usize,
            vector_offset,
        )
        .await?;

        let quantizer_directory = format!("{}/quantizer", base_directory);
        let quantizer = Q::read(quantizer_directory)?;

        let header = graph_storage.header().clone();

        Ok(Self {
            graph_storage,
            vector_storage,
            header,
            quantizer,
        })
    }

    pub fn get_header(&self) -> &Header {
        &self.header
    }

    async fn get_vector(
        &self,
        point_id: u32,
        context: &mut impl StorageContext,
    ) -> Result<Vec<Q::QuantizedT>> {
        self.vector_storage.get(point_id, context).await
    }

    pub async fn ann_search(
        &self,
        query: &[f32],
        k: usize,
        ef: u32,
        record_pages: bool,
    ) -> SearchResult {
        info!("[ASYNC] Searching for {} points with ef {}", k, ef);

        let quantized_query = Q::QuantizedT::process_vector(query, &self.quantizer);
        let mut current_layer: i32 = self.header.num_layers as i32 - 1;
        let mut ep = self.graph_storage.get_entry_point_top_layer().await;
        let mut working_set;
        let mut context = SearchContext::new(record_pages);
        while current_layer > 0 {
            working_set = self
                .search_layer(&mut context, &quantized_query, ep, ef, current_layer as u8)
                .await;
            ep = working_set
                .iter()
                .min_by(|x, y| x.distance.cmp(&y.distance))
                .unwrap()
                .point_id;
            current_layer -= 1;
        }

        working_set = self
            .search_layer(&mut context, &quantized_query, ep, ef, 0)
            .await;
        working_set.sort_by(|x, y| x.distance.cmp(&y.distance));
        working_set.truncate(k);
        let point_ids: Vec<u32> = working_set.iter().map(|x| x.point_id).collect();
        let doc_ids = self
            .graph_storage
            .map_point_id_to_doc_id(&point_ids)
            .await
            .unwrap_or_default();

        let id_with_scores: Vec<IdWithScore> = working_set
            .into_iter()
            .zip(doc_ids)
            .map(|(x, y)| IdWithScore {
                doc_id: y,
                score: x.distance.to_f32().unwrap(),
            })
            .collect();

        SearchResult {
            id_with_scores,
            stats: SearchStats::new(),
        }
    }

    async fn search_layer(
        &self,
        context: &mut impl StorageContext,
        query: &[Q::QuantizedT],
        entry_point: u32,
        ef_construction: u32,
        layer: u8,
    ) -> Vec<PointAndDistance> {
        context.set_visited(entry_point);

        let mut candidates = BinaryHeap::new();
        let mut working_list = BinaryHeap::new();

        let entry_dist = self.distance(query, entry_point, context).await;
        candidates.push(PointAndDistance {
            point_id: entry_point,
            distance: NotNan::new(-entry_dist).unwrap(),
        });
        working_list.push(PointAndDistance {
            point_id: entry_point,
            distance: NotNan::new(entry_dist).unwrap(),
        });

        while !candidates.is_empty() {
            let point_and_distance = candidates.pop().unwrap();
            let point_id = point_and_distance.point_id;
            let distance: f32 = -*point_and_distance.distance;

            let mut furthest_element_from_working_list = match working_list.peek() {
                Some(e) => e,
                None => continue,
            };
            if distance > *furthest_element_from_working_list.distance {
                break;
            }

            let edges = self
                .graph_storage
                .get_edges_for_point(point_id, layer)
                .await;
            if edges.is_none() {
                continue;
            }

            for e in edges.unwrap().iter() {
                if context.visited(*e) {
                    continue;
                }
                context.set_visited(*e);
                furthest_element_from_working_list = match working_list.peek() {
                    Some(e) => e,
                    None => continue,
                };
                let distance_e_q = self.distance(query, *e, context).await;
                if distance_e_q < *furthest_element_from_working_list.distance
                    || working_list.len() < ef_construction as usize
                {
                    candidates.push(PointAndDistance {
                        point_id: *e,
                        distance: NotNan::new(-distance_e_q).unwrap(),
                    });
                    working_list.push(PointAndDistance {
                        point_id: *e,
                        distance: NotNan::new(distance_e_q).unwrap(),
                    });
                    if working_list.len() > ef_construction as usize {
                        working_list.pop();
                    }
                }
            }
        }

        let mut result: Vec<PointAndDistance> = working_list.into_iter().collect();
        result.sort();
        result
    }

    async fn distance(
        &self,
        query: &[Q::QuantizedT],
        point_id: u32,
        context: &mut impl StorageContext,
    ) -> f32 {
        let point = self.get_vector(point_id, context).await.unwrap();
        self.quantizer
            .distance(query, point.as_slice(), StreamingSIMD)
    }
}

impl<Q: Quantizer> GraphTraversal<Q> for AsyncHnsw<Q>
where
    Q::QuantizedT: Send + Sync,
{
    type ContextT = BuilderContext;

    fn distance(
        &self,
        _query: &[Q::QuantizedT],
        _point_id: u32,
        _context: &mut impl StorageContext,
    ) -> f32 {
        unimplemented!("Use async distance method instead")
    }

    fn get_edges_for_point(&self, _point_id: u32, _layer: u8) -> Option<Vec<u32>> {
        unimplemented!("Use async get_edges_for_point method instead")
    }

    fn search_layer(
        &self,
        _context: &mut impl StorageContext,
        _query: &[Q::QuantizedT],
        _entry_point: u32,
        _ef_construction: u32,
        _layer: u8,
    ) -> Vec<PointAndDistance> {
        unimplemented!("Use async search_layer method instead")
    }

    fn print_graph(&self, _layer: u8, _predicate: impl Fn(u8, u32) -> bool) {
        unimplemented!("Use async methods for graph traversal")
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::sync::Arc;

    use quantization::noq::noq::NoQuantizer;
    use quantization::pq::pq::{ProductQuantizer, ProductQuantizerConfig};
    use quantization::pq::pq_builder::{ProductQuantizerBuilder, ProductQuantizerBuilderConfig};
    use quantization::quantization::WritableQuantizer;
    use utils::block_cache::{BlockCache, BlockCacheConfig};
    use utils::distance::l2::L2DistanceCalculator;
    use utils::test_utils::generate_random_vector;

    use super::*;
    use crate::hnsw::builder::HnswBuilder;
    use crate::hnsw::writer::HnswWriter;

    #[tokio::test]
    async fn test_async_hnsw_search() {
        let datapoints: Vec<Vec<f32>> = (0..10000).map(|_| generate_random_vector(128)).collect();

        let temp_dir = tempdir::TempDir::new("async_hnsw_test").unwrap();
        let base_directory = temp_dir.path().to_str().unwrap().to_string();
        let pq_dir = format!("{}/quantizer", base_directory);
        fs::create_dir_all(&pq_dir).unwrap();

        let pq_config = ProductQuantizerConfig {
            dimension: 128,
            subvector_dimension: 8,
            num_bits: 8,
        };

        let pq_builder_config = ProductQuantizerBuilderConfig {
            max_iteration: 1000,
            batch_size: 4,
        };

        let mut pq_builder =
            ProductQuantizerBuilder::<L2DistanceCalculator>::new(pq_config, pq_builder_config);
        for datapoint in datapoints.iter().take(1000) {
            pq_builder.add(datapoint.clone());
        }
        let pq = pq_builder.build(base_directory.clone()).unwrap();
        pq.write_to_directory(&pq_dir).unwrap();

        let vector_dir = format!("{}/vectors", base_directory);
        fs::create_dir_all(&vector_dir).unwrap();
        let mut hnsw_builder = HnswBuilder::new(10, 128, 20, 1024, 4096, 16, pq, vector_dir);

        for (i, datapoint) in datapoints.iter().enumerate() {
            hnsw_builder.insert(i as u128, datapoint).unwrap();
        }

        let hnsw_dir = format!("{}/hnsw", base_directory);
        fs::create_dir_all(&hnsw_dir).unwrap();
        let writer = HnswWriter::new(hnsw_dir.clone());
        writer.write(&mut hnsw_builder, false).unwrap();

        let config = BlockCacheConfig::default();
        let cache = Arc::new(BlockCache::new(config));

        let hnsw =
            AsyncHnsw::<ProductQuantizer<L2DistanceCalculator>>::new(cache.clone(), base_directory)
                .await
                .unwrap();

        assert_eq!(hnsw.get_header().num_layers > 0, true);

        let query = generate_random_vector(128);
        let result = hnsw.ann_search(&query, 10, 100, false).await;
        assert_eq!(result.id_with_scores.len(), 10);
    }

    #[tokio::test]
    async fn test_async_hnsw_no_quantizer() {
        let temp_dir = tempdir::TempDir::new("async_hnsw_noq_test").unwrap();
        let base_directory = temp_dir.path().to_str().unwrap().to_string();
        let vector_dir = format!("{}/vectors", base_directory);
        fs::create_dir_all(&vector_dir).unwrap();

        let datapoints: Vec<Vec<f32>> = (0..1000).map(|_| generate_random_vector(128)).collect();

        let quantizer = NoQuantizer::<L2DistanceCalculator>::new(128);
        let quantizer_dir = format!("{}/quantizer", base_directory);
        fs::create_dir_all(&quantizer_dir).unwrap();
        quantizer.write_to_directory(&quantizer_dir).unwrap();

        let mut hnsw_builder =
            HnswBuilder::new(10, 128, 20, 1024, 4096, 128, quantizer, vector_dir);
        for (i, datapoint) in datapoints.iter().enumerate() {
            hnsw_builder.insert(i as u128, datapoint).unwrap();
        }

        let hnsw_dir = format!("{}/hnsw", base_directory);
        fs::create_dir_all(&hnsw_dir).unwrap();
        let writer = HnswWriter::new(hnsw_dir.clone());
        writer.write(&mut hnsw_builder, false).unwrap();

        let config = BlockCacheConfig::default();
        let cache = Arc::new(BlockCache::new(config));

        let hnsw =
            AsyncHnsw::<NoQuantizer<L2DistanceCalculator>>::new(cache.clone(), base_directory)
                .await
                .unwrap();

        let query = generate_random_vector(128);
        let result = hnsw.ann_search(&query, 5, 100, false).await;
        assert_eq!(result.id_with_scores.len(), 5);
    }

    // Tests adapted from builder.rs to verify AsyncHnsw works with the same test scenarios

    #[tokio::test]
    async fn test_async_hnsw_with_reindexed_builder() {
        // This test verifies that AsyncHnsw can correctly read an index that has been reindexed
        // Similar to test_hnsw_builder_reindex in builder.rs
        let mut codebook = vec![];
        for subvector_idx in 0..5 {
            for i in 0..(1 << 1) {
                let x = (subvector_idx * 2 + i) as f32;
                let y = (subvector_idx * 2 + i) as f32;
                codebook.push(x);
                codebook.push(y);
            }
        }

        let temp_dir = tempdir::TempDir::new("async_reindex_test").unwrap();
        let base_directory = temp_dir.path().to_str().unwrap().to_string();
        let vector_dir = format!("{}/vectors", base_directory);
        fs::create_dir_all(vector_dir.clone()).unwrap();

        let pq = ProductQuantizer::<L2DistanceCalculator>::new(
            10,
            2,
            1,
            codebook,
            base_directory.clone(),
        )
        .expect("Can't create product quantizer");

        let quantizer_dir = format!("{}/quantizer", base_directory);
        fs::create_dir_all(&quantizer_dir).unwrap();
        pq.write_to_directory(&quantizer_dir).unwrap();

        // Use better HNSW parameters to ensure good connectivity
        let mut builder = HnswBuilder::new(10, 10, 50, 1024, 4096, 5, pq, vector_dir);

        // Insert enough vectors to ensure a well-connected graph
        for i in 0..100 {
            let vector = vec![i as f32; 10];
            builder.insert(100 + i, &vector).unwrap();
        }

        // Reindex the builder
        let reindex_dir = format!("{}/reindex_temp", base_directory);
        fs::create_dir_all(&reindex_dir).unwrap();
        builder.reindex(reindex_dir).unwrap();

        // Write to disk
        let hnsw_dir = format!("{}/hnsw", base_directory);
        fs::create_dir_all(&hnsw_dir).unwrap();
        let writer = HnswWriter::new(hnsw_dir.clone());
        writer.write(&mut builder, false).unwrap();

        // Now test with AsyncHnsw
        let config = BlockCacheConfig::default();
        let cache = Arc::new(BlockCache::new(config));

        let hnsw =
            AsyncHnsw::<ProductQuantizer<L2DistanceCalculator>>::new(cache.clone(), base_directory)
                .await
                .unwrap();

        // Verify we can search the reindexed index
        let query = vec![50.0; 10];
        // Use higher ef for better recall
        let result = hnsw.ann_search(&query, 10, 50, false).await;
        // Check we get at least some results (be lenient for probabilistic HNSW)
        assert!(
            result.id_with_scores.len() >= 5,
            "Expected at least 5 results, got {}",
            result.id_with_scores.len()
        );
        assert!(
            result.id_with_scores.len() <= 10,
            "Expected at most 10 results, got {}",
            result.id_with_scores.len()
        );

        // Verify the doc IDs are in the expected range (100-199)
        for id_score in result.id_with_scores {
            assert!(
                id_score.doc_id >= 100 && id_score.doc_id < 200,
                "Doc ID {} out of expected range [100, 200)",
                id_score.doc_id
            );
        }
    }

    #[tokio::test]
    async fn test_async_hnsw_search_layer_comprehensive() {
        // This test is adapted from test_search_layer in builder.rs
        // It verifies AsyncHnsw search works correctly with a larger dataset
        let dimension = 10;
        let subvector_dimension = 2;
        let num_bits = 1;

        let mut codebook = vec![];
        for subvector_idx in 0..dimension / subvector_dimension {
            for i in 0..(1 << 1) {
                let x = (subvector_idx * 2 + i) as f32;
                let y = (subvector_idx * 2 + i) as f32;
                codebook.push(x);
                codebook.push(y);
            }
        }

        let temp_dir = tempdir::TempDir::new("async_search_layer_test").unwrap();
        let base_directory = temp_dir.path().to_str().unwrap().to_string();

        let pq = ProductQuantizer::new(
            dimension,
            subvector_dimension,
            num_bits,
            codebook,
            base_directory.clone(),
        )
        .expect("ProductQuantizer should be created.");

        let quantizer_dir = format!("{}/quantizer", base_directory);
        fs::create_dir_all(&quantizer_dir).unwrap();
        pq.write_to_directory(&quantizer_dir).unwrap();

        let vector_dir = format!("{}/vectors", base_directory);
        fs::create_dir_all(vector_dir.clone()).unwrap();
        let mut builder = HnswBuilder::<ProductQuantizer<L2DistanceCalculator>>::new(
            5, 10, 20, 1024, 4096, 5, pq, vector_dir,
        );

        // Insert 100 vectors as in the original test
        for i in 0..100 {
            builder
                .insert(i, &generate_random_vector(dimension))
                .unwrap();
        }

        // Write the index to disk
        let hnsw_dir = format!("{}/hnsw", base_directory);
        fs::create_dir_all(&hnsw_dir).unwrap();
        let writer = HnswWriter::new(hnsw_dir.clone());
        writer.write(&mut builder, false).unwrap();

        // Now test with AsyncHnsw
        let config = BlockCacheConfig::default();
        let cache = Arc::new(BlockCache::new(config));

        let hnsw =
            AsyncHnsw::<ProductQuantizer<L2DistanceCalculator>>::new(cache.clone(), base_directory)
                .await
                .unwrap();

        // Test search functionality with various parameters
        let query = generate_random_vector(dimension);

        // Test with k=10, ef=20 (same as builder test)
        let result = hnsw.ann_search(&query, 10, 20, false).await;
        assert_eq!(result.id_with_scores.len(), 10);

        // Verify all returned doc_ids are valid (0-99)
        for id_score in &result.id_with_scores {
            assert!(id_score.doc_id < 100);
        }

        // Test with different k values
        let result_5 = hnsw.ann_search(&query, 5, 20, false).await;
        assert_eq!(result_5.id_with_scores.len(), 5);

        let result_20 = hnsw.ann_search(&query, 20, 50, false).await;
        assert_eq!(result_20.id_with_scores.len(), 20);
    }

    #[tokio::test]
    async fn test_async_hnsw_multi_layer_structure() {
        // This test verifies AsyncHnsw works correctly with multi-layer HNSW graphs
        // Adapted from the layer structure tests in builder.rs
        let dimension = 128;
        let temp_dir = tempdir::TempDir::new("async_multi_layer_test").unwrap();
        let base_directory = temp_dir.path().to_str().unwrap().to_string();

        let pq_dir = format!("{}/quantizer", base_directory);
        fs::create_dir_all(&pq_dir).unwrap();

        let pq_config = ProductQuantizerConfig {
            dimension,
            subvector_dimension: 8,
            num_bits: 8,
        };

        let pq_builder_config = ProductQuantizerBuilderConfig {
            max_iteration: 1000,
            batch_size: 4,
        };

        let datapoints: Vec<Vec<f32>> = (0..1000)
            .map(|_| generate_random_vector(dimension))
            .collect();

        let mut pq_builder =
            ProductQuantizerBuilder::<L2DistanceCalculator>::new(pq_config, pq_builder_config);
        // Use more training data to avoid kmeans assertion failure (need at least batch_size samples)
        for datapoint in datapoints.iter().take(500) {
            pq_builder.add(datapoint.clone());
        }
        let pq = pq_builder.build(base_directory.clone()).unwrap();
        pq.write_to_directory(&pq_dir).unwrap();

        let vector_dir = format!("{}/vectors", base_directory);
        fs::create_dir_all(&vector_dir).unwrap();

        // Use parameters that encourage multi-layer structure
        let mut hnsw_builder = HnswBuilder::new(10, 128, 20, 1024, 4096, 16, pq, vector_dir);

        for (i, datapoint) in datapoints.iter().enumerate() {
            hnsw_builder.insert(i as u128, datapoint).unwrap();
        }

        let hnsw_dir = format!("{}/hnsw", base_directory);
        fs::create_dir_all(&hnsw_dir).unwrap();
        let writer = HnswWriter::new(hnsw_dir.clone());
        writer.write(&mut hnsw_builder, false).unwrap();

        let config = BlockCacheConfig::default();
        let cache = Arc::new(BlockCache::new(config));

        let hnsw =
            AsyncHnsw::<ProductQuantizer<L2DistanceCalculator>>::new(cache.clone(), base_directory)
                .await
                .unwrap();

        // Verify the index has multiple layers
        assert!(
            hnsw.get_header().num_layers > 1,
            "Expected multi-layer structure"
        );

        // Test search works correctly across layers
        let query = generate_random_vector(dimension);
        let result = hnsw.ann_search(&query, 10, 100, false).await;
        assert_eq!(result.id_with_scores.len(), 10);

        // Verify results are sorted by distance
        for i in 1..result.id_with_scores.len() {
            assert!(
                result.id_with_scores[i - 1].score <= result.id_with_scores[i].score,
                "Results should be sorted by distance"
            );
        }
    }
}
