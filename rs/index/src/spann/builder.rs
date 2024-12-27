use anyhow::{Ok, Result};
use quantization::noq::noq::NoQuantizer;

use crate::hnsw::builder::HnswBuilder;
use crate::ivf::builder::{IvfBuilder, IvfBuilderConfig};

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
    pub num_data_points: usize,
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

pub struct SpannBuilder {
    pub config: SpannBuilderConfig,
    pub ivf_builder: IvfBuilder,
    pub centroid_builder: HnswBuilder<NoQuantizer>,
}

impl SpannBuilder {
    pub fn new(config: SpannBuilderConfig) -> Result<Self> {
        let ivf_builder = IvfBuilder::new(IvfBuilderConfig {
            max_iteration: config.max_iteration,
            batch_size: config.batch_size,
            num_clusters: config.num_clusters,
            num_data_points: config.num_data_points,
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

        let centroid_storage = self.ivf_builder.centroids();
        let num_centroids = centroid_storage.len();

        for i in 0..num_centroids {
            self.centroid_builder
                .insert(i as u64, &centroid_storage.get(i as u32).unwrap())?;
        }

        Ok(())
    }
}
