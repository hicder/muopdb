// TODO(hicder): support more quantizers
#[derive(Debug, Clone, Default)]
pub enum QuantizerType {
    #[default]
    ProductQuantizer,
}

#[derive(Debug, Clone, Default)]
pub struct IndexWriterConfig {
    pub output_path: String,

    // HNSW parameters
    pub num_layers: u8,
    pub max_num_neighbors: usize,
    pub ef_construction: u32,

    // Quantizer parameters
    pub quantizer_type: QuantizerType,

    pub dimension: usize,
    pub subvector_dimension: usize,
    pub num_bits: u8,
    pub num_training_rows: usize,

    // Quantizer builder parameters
    pub max_iteration: usize,
    pub batch_size: usize,

    // Vector storage parameters
    pub max_memory_size: usize,
    pub file_size: usize,
}
