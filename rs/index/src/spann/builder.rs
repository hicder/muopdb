use anyhow::{Ok, Result};
use config::collection::CollectionConfig;
use config::enums::{IntSeqEncodingType, QuantizerType};
use log::debug;
use quantization::noq::noq::NoQuantizer;
use serde::{Deserialize, Serialize};
use utils::distance::l2::L2DistanceCalculator;

use crate::hnsw::builder::HnswBuilder;
use crate::ivf::builder::{IvfBuilder, IvfBuilderConfig};
use crate::utils::SearchContext;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct SpannBuilderConfig {
    // For centroids
    pub centroids_max_neighbors: usize,
    pub centroids_max_layers: u8,
    pub centroids_ef_construction: u32,
    pub centroids_vector_storage_memory_size: usize,
    pub centroids_vector_storage_file_size: usize,
    pub centroids_clustering_tolerance: f32,
    pub num_features: usize,

    // For quantization
    pub pq_subvector_dimension: usize,
    pub pq_num_bits: usize,
    pub pq_max_iteration: usize,
    pub pq_batch_size: usize,
    pub pq_num_training_rows: usize,
    pub quantizer_type: QuantizerType,

    // For posting lists
    pub ivf_num_clusters: usize,
    pub ivf_num_data_points_for_clustering: usize,
    pub ivf_max_clusters_per_vector: usize,
    pub ivf_distance_threshold: f32, // Threshold to add a vector to more than one cluster
    pub posting_list_encoding_type: IntSeqEncodingType,

    // Parameters for storages
    pub ivf_base_directory: String,
    pub ivf_vector_storage_memory_size: usize,
    pub ivf_vector_storage_file_size: usize,

    // Parameters for clustering
    pub ivf_max_posting_list_size: usize,

    // Optimization parameters
    pub reindex: bool,
}

impl SpannBuilderConfig {
    pub fn from_collection_config(
        collection_config: &CollectionConfig,
        base_directory: String,
    ) -> Self {
        Self {
            centroids_max_neighbors: collection_config.centroids_max_neighbors,
            centroids_max_layers: collection_config.centroids_max_layers,
            centroids_ef_construction: collection_config.centroids_ef_construction,
            centroids_vector_storage_memory_size: collection_config
                .centroids_builder_vector_storage_memory_size,
            centroids_vector_storage_file_size: collection_config
                .centroids_builder_vector_storage_file_size,
            num_features: collection_config.num_features,

            pq_subvector_dimension: collection_config.product_quantization_subvector_dimension,
            pq_num_bits: collection_config.product_quantization_num_bits,
            pq_max_iteration: collection_config.product_quantization_max_iteration,
            pq_batch_size: collection_config.product_quantization_batch_size,
            pq_num_training_rows: collection_config.product_quantization_num_training_rows,
            quantizer_type: collection_config.quantization_type.clone(),

            ivf_num_clusters: collection_config.initial_num_centroids,
            ivf_num_data_points_for_clustering: collection_config.num_data_points_for_clustering,
            ivf_max_clusters_per_vector: collection_config.max_clusters_per_vector,
            ivf_distance_threshold: collection_config.clustering_distance_threshold_pct,
            posting_list_encoding_type: collection_config.posting_list_encoding_type.clone(),

            ivf_base_directory: base_directory,
            ivf_vector_storage_memory_size: collection_config
                .posting_list_builder_vector_storage_memory_size,
            ivf_vector_storage_file_size: collection_config
                .posting_list_builder_vector_storage_file_size,

            centroids_clustering_tolerance: collection_config
                .posting_list_kmeans_unbalanced_penalty,
            ivf_max_posting_list_size: collection_config.max_posting_list_size,

            reindex: collection_config.reindex,
        }
    }
}

impl Default for SpannBuilderConfig {
    fn default() -> Self {
        Self {
            centroids_max_neighbors: 10,
            centroids_max_layers: 2,
            centroids_ef_construction: 100,
            centroids_vector_storage_memory_size: 1024,
            centroids_vector_storage_file_size: 1024,
            num_features: 768,

            pq_subvector_dimension: 8,
            pq_num_bits: 8,
            pq_max_iteration: 1000,
            pq_batch_size: 4,
            pq_num_training_rows: 10000,
            quantizer_type: QuantizerType::NoQuantizer,

            ivf_num_clusters: 10,
            ivf_num_data_points_for_clustering: 1000,
            ivf_max_clusters_per_vector: 1,
            ivf_distance_threshold: 0.1,
            posting_list_encoding_type: IntSeqEncodingType::PlainEncoding,

            ivf_base_directory: "./".to_string(),
            ivf_vector_storage_memory_size: 1024,
            ivf_vector_storage_file_size: 1024,

            centroids_clustering_tolerance: 0.1,
            ivf_max_posting_list_size: usize::MAX,

            reindex: true,
        }
    }
}

pub struct SpannBuilder {
    pub config: SpannBuilderConfig,
    pub ivf_builder: IvfBuilder<L2DistanceCalculator>,
    pub centroid_builder: HnswBuilder<NoQuantizer<L2DistanceCalculator>>,
}

impl SpannBuilder {
    pub fn new(config: SpannBuilderConfig) -> Result<Self> {
        let ivf_builder = IvfBuilder::<L2DistanceCalculator>::new(IvfBuilderConfig {
            max_iteration: config.pq_max_iteration,
            batch_size: config.pq_batch_size,
            num_clusters: config.ivf_num_clusters,
            num_data_points_for_clustering: config.ivf_num_data_points_for_clustering,
            max_clusters_per_vector: config.ivf_max_clusters_per_vector,
            distance_threshold: config.ivf_distance_threshold,
            base_directory: config.ivf_base_directory.clone(),
            memory_size: config.ivf_vector_storage_memory_size,
            file_size: config.ivf_vector_storage_file_size,
            num_features: config.num_features,
            tolerance: config.centroids_clustering_tolerance,
            max_posting_list_size: config.ivf_max_posting_list_size,
        })?;

        let centroid_directory = format!("{}/centroids", config.ivf_base_directory.clone());
        std::fs::create_dir_all(&centroid_directory)?;

        let hnsw_directory = format!("{}/hnsw", centroid_directory);
        std::fs::create_dir_all(&hnsw_directory)?;

        let centroid_quantizer = NoQuantizer::new(config.num_features);
        let centroid_builder = HnswBuilder::new(
            config.centroids_max_neighbors,
            config.centroids_max_layers,
            config.centroids_ef_construction,
            config.ivf_vector_storage_memory_size,
            config.ivf_vector_storage_file_size,
            config.num_features,
            centroid_quantizer,
            hnsw_directory,
        );

        Ok(Self {
            config,
            ivf_builder,
            centroid_builder,
        })
    }

    #[allow(unused_variables)]
    pub fn add(&mut self, doc_id: u128, data: &[f32]) -> Result<()> {
        self.ivf_builder.add_vector(doc_id, data)
    }

    pub fn build(&mut self) -> Result<()> {
        self.ivf_builder.build()?;
        debug!("Finish building IVF index");

        let centroid_storage = self.ivf_builder.centroids();
        let num_centroids = centroid_storage.borrow().num_vectors();
        let mut search_context = SearchContext::new(false);

        for i in 0..num_centroids {
            self.centroid_builder.insert(
                i as u128,
                &centroid_storage
                    .borrow()
                    .get(i as u32, &mut search_context)
                    .unwrap(),
            )?;
        }
        debug!("Finish building centroids");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use config::collection::CollectionConfig;

    use crate::spann::builder::SpannBuilderConfig;

    #[test]
    fn test_read_write_config() {
        use std::fs::File;

        let temp_dir = tempdir::TempDir::new("test_read_write_config").unwrap();
        let base_directory = temp_dir.path().to_str().unwrap().to_string();

        // Write the collection config
        let collection_config_path = format!("{}/collection_config.json", base_directory);
        let collection_config = SpannBuilderConfig::from_collection_config(
            &CollectionConfig::default_test_config(),
            base_directory.clone(),
        );
        std::fs::create_dir_all(&base_directory).unwrap();
        let mut file = File::create(collection_config_path.clone()).unwrap();
        serde_json::to_writer(&mut file, &collection_config).unwrap();

        // Read the collection config
        let read_collection_config: SpannBuilderConfig =
            serde_json::from_reader(File::open(collection_config_path).unwrap()).unwrap();
        assert_eq!(collection_config, read_collection_config);
    }
}
