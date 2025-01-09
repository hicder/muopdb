use serde::{Deserialize, Serialize};

// TODO(hicder): support more quantizers
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub enum QuantizerType {
    ProductQuantizer,
    #[default]
    NoQuantizer,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub enum DistanceType {
    DotProduct,
    #[default]
    L2,
}

// TODO(tyb): support more encoding
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub enum IntSeqEncodingType {
    EliasFano,
    #[default]
    PlainEncoding,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub enum IndexType {
    Hnsw,
    Ivf,
    #[default]
    Spann,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BaseConfig {
    pub output_path: String,
    pub dimension: usize,
    pub reindex: bool,

    // Vector storage parameters
    pub max_memory_size: usize,
    pub file_size: usize,

    pub index_type: IndexType,
    pub index_distance_type: DistanceType,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QuantizerConfig {
    // Quantizer parameters
    pub quantizer_type: QuantizerType,
    pub subvector_dimension: usize,
    pub num_bits: u8,
    pub num_training_rows: usize,
    pub quantizer_distance_type: DistanceType,

    // Quantizer builder parameters
    pub max_iteration: usize,
    pub batch_size: usize,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HnswConfig {
    pub num_layers: u8,
    pub max_num_neighbors: usize,
    pub ef_construction: u32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IvfConfig {
    // IVF parameters
    pub num_clusters: usize,
    pub num_data_points: usize,
    pub max_clusters_per_vector: usize,
    pub distance_threshold: f32,
    pub posting_list_encoding_type: IntSeqEncodingType,

    // KMeans training parameters
    pub max_iteration: usize,
    pub batch_size: usize,
    pub tolerance: f32,
    pub max_posting_list_size: usize,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IvfConfigWithBase {
    pub base_config: BaseConfig,
    pub quantizer_config: QuantizerConfig,
    pub ivf_config: IvfConfig,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HnswConfigWithBase {
    pub base_config: BaseConfig,
    pub quantizer_config: QuantizerConfig,
    pub hnsw_config: HnswConfig,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SpannConfigWithBase {
    pub base_config: BaseConfig,
    pub quantizer_config: QuantizerConfig,
    pub hnsw_config: HnswConfig,
    pub ivf_config: IvfConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IndexWriterConfig {
    Hnsw(HnswConfigWithBase),
    Ivf(IvfConfigWithBase),
    Spann(SpannConfigWithBase),
}

impl Default for IndexWriterConfig {
    fn default() -> Self {
        IndexWriterConfig::Hnsw(HnswConfigWithBase::default())
    }
}
