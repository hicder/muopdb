// Script to write the index writer config. Currently it's used to generate the
// config for hnsw-ivf index for sift128 dataset.
use std::fs::File;
use std::io::Write;

use index_writer::config::{BaseConfig, HnswConfig, IvfConfig, QuantizerType, SpannConfigWithBase};

fn main() -> std::io::Result<()> {
    let mut base_config = BaseConfig::default();
    base_config.reindex = false;
    base_config.dimension = 128;
    base_config.output_path = "NONE".to_string();
    base_config.max_memory_size = 1024 * 1024 * 1024; // 1 GB
    base_config.file_size = 1024 * 1024 * 1024; // 1 GB
    base_config.index_type = index_writer::config::IndexType::Spann;

    let mut ivf_config = IvfConfig::default();
    ivf_config.num_clusters = 5;
    ivf_config.num_data_points = 10000;
    ivf_config.max_clusters_per_vector = 1;
    ivf_config.max_iteration = 1000;
    ivf_config.batch_size = 4;
    ivf_config.max_posting_list_size = 1000;

    let mut hnsw_config = HnswConfig::default();
    hnsw_config.quantizer_type = QuantizerType::NoQuantizer;
    hnsw_config.num_layers = 4;
    hnsw_config.max_num_neighbors = 32;
    hnsw_config.ef_construction = 200;

    let mut config = SpannConfigWithBase::default();
    config.base_config = base_config;
    config.ivf_config = ivf_config;
    config.hnsw_config = hnsw_config;

    let mut file = File::create("/tmp/index_writer_config.yaml")?;
    file.write_all(serde_yaml::to_string(&config).unwrap().as_bytes())?;
    Ok(())
}
