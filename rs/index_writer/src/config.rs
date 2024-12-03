use serde::{Deserialize, Serialize};

// TODO(hicder): support more quantizers
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum QuantizerType {
    #[default]
    ProductQuantizer,
    NoQuantizer,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BaseConfig {
    pub output_path: String,
    pub dimension: usize,

    // Vector storage parameters
    pub max_memory_size: usize,
    pub file_size: usize,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HnswConfig {
    // HNSW parameters
    pub num_layers: u8,
    pub max_num_neighbors: usize,
    pub ef_construction: u32,
    pub reindex: bool,

    // Quantizer parameters
    pub quantizer_type: QuantizerType,
    pub subvector_dimension: usize,
    pub num_bits: u8,
    pub num_training_rows: usize,

    // Quantizer builder parameters
    pub max_iteration: usize,
    pub batch_size: usize,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IvfConfig {
    // IVF parameters
    pub num_clusters: usize,
    pub num_data_points: usize,
    pub max_clusters_per_vector: usize,

    // KMeans training parameters
    pub max_iteration: usize,
    pub batch_size: usize,
}


#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IvfConfigWithBase {
    pub base_config: BaseConfig,
    pub ivf_config: IvfConfig,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HnswConfigWithBase {
    pub base_config: BaseConfig,
    pub hnsw_config: HnswConfig,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HnswIvfConfig {
    pub base_config: BaseConfig,
    pub hnsw_config: HnswConfig,
    pub ivf_config: IvfConfig,
}

#[derive(Debug, Clone)]
pub enum IndexWriterConfig {
    Hnsw(HnswConfigWithBase),
    Ivf(IvfConfigWithBase),
    HnswIvf(HnswIvfConfig),
}

impl Default for IndexWriterConfig {
    fn default() -> Self {
        IndexWriterConfig::Hnsw(HnswConfigWithBase::default())
    }
}
