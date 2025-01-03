use anyhow::{Ok, Result};
use log::debug;
use quantization::noq::noq::NoQuantizer;
use serde::{Deserialize, Serialize};
use utils::distance::l2::L2DistanceCalculator;

use crate::hnsw::builder::HnswBuilder;
use crate::ivf::builder::{IvfBuilder, IvfBuilderConfig};

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct SpannBuilderConfig {
    // For centroids
    pub max_neighbors: usize,
    pub max_layers: u8,
    pub ef_construction: u32,
    pub vector_storage_memory_size: usize,
    pub vector_storage_file_size: usize,
    pub num_features: usize,

    // For posting lists
    pub max_iteration: usize,
    pub batch_size: usize,
    pub num_clusters: usize,
    pub num_data_points_for_clustering: usize,
    pub max_clusters_per_vector: usize,
    // Threshold to add a vector to more than one cluster
    pub distance_threshold: f32,

    // Parameters for storages
    pub base_directory: String,
    pub memory_size: usize,
    pub file_size: usize,

    // Parameters for clustering.
    pub tolerance: f32,
    pub max_posting_list_size: usize,

    pub reindex: bool,
}

impl Default for SpannBuilderConfig {
    fn default() -> Self {
        Self {
            max_neighbors: 10,
            max_layers: 2,
            ef_construction: 100,
            vector_storage_memory_size: 1024,
            vector_storage_file_size: 1024,
            num_features: 768,
            max_iteration: 1000,
            batch_size: 4,
            num_clusters: 10,
            num_data_points_for_clustering: 1000,
            max_clusters_per_vector: 1,
            distance_threshold: 0.1,
            base_directory: "./".to_string(),
            memory_size: 1024,
            file_size: 1024,
            tolerance: 0.1,
            max_posting_list_size: usize::MAX,
            reindex: true,
        }
    }
}

pub struct SpannBuilder {
    pub config: SpannBuilderConfig,
    pub ivf_builder: IvfBuilder<L2DistanceCalculator>,
    pub centroid_builder: HnswBuilder<NoQuantizer>,
}

impl SpannBuilder {
    pub fn new(config: SpannBuilderConfig) -> Result<Self> {
        let ivf_builder = IvfBuilder::<L2DistanceCalculator>::new(IvfBuilderConfig {
            max_iteration: config.max_iteration,
            batch_size: config.batch_size,
            num_clusters: config.num_clusters,
            num_data_points_for_clustering: config.num_data_points_for_clustering,
            max_clusters_per_vector: config.max_clusters_per_vector,
            distance_threshold: config.distance_threshold,
            base_directory: config.base_directory.clone(),
            memory_size: config.memory_size,
            file_size: config.file_size,
            num_features: config.num_features,
            tolerance: config.tolerance,
            max_posting_list_size: config.max_posting_list_size,
        })?;

        let centroid_directory = format!("{}/centroids", config.base_directory.clone());
        std::fs::create_dir_all(&centroid_directory)?;

        let hnsw_directory = format!("{}/hnsw", centroid_directory);
        std::fs::create_dir_all(&hnsw_directory)?;

        let centroid_quantizer = NoQuantizer::new(config.num_features);
        let centroid_builder = HnswBuilder::new(
            config.max_neighbors,
            config.max_layers,
            config.ef_construction,
            config.memory_size,
            config.file_size,
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
    pub fn add(&mut self, doc_id: u64, data: &[f32]) -> Result<()> {
        self.ivf_builder.add_vector(doc_id, data)
    }

    pub fn build(&mut self) -> Result<()> {
        self.ivf_builder.build()?;
        debug!("Finish building IVF index");

        let centroid_storage = self.ivf_builder.centroids();
        let num_centroids = centroid_storage.borrow().len();

        for i in 0..num_centroids {
            self.centroid_builder
                .insert(i as u64, &centroid_storage.borrow().get(i as u32).unwrap())?;
        }
        debug!("Finish building centroids");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::spann::builder::SpannBuilderConfig;

    #[test]
    fn test_read_write_config() {
        use std::fs::File;

        let temp_dir = tempdir::TempDir::new("test_read_write_config").unwrap();
        let base_directory = temp_dir.path().to_str().unwrap().to_string();

        // Write the collection config
        let collection_config_path = format!("{}/collection_config.json", base_directory);
        let collection_config = SpannBuilderConfig::default();
        std::fs::create_dir_all(&base_directory).unwrap();
        let mut file = File::create(collection_config_path.clone()).unwrap();
        serde_json::to_writer(&mut file, &collection_config).unwrap();

        // Read the collection config
        let read_collection_config: SpannBuilderConfig =
            serde_json::from_reader(File::open(collection_config_path).unwrap()).unwrap();
        assert_eq!(collection_config, read_collection_config);
    }
}
