// Script to write the index writer config. Currently it's used to generate the
// config for hnsw-ivf index for sift128 dataset.
use std::io::Write;

use anyhow::Result;
use clap::Parser;
use index_writer::config::{
    BaseConfig, HnswConfig, HnswConfigWithBase, IndexWriterConfig, IntSeqEncodingType, IvfConfig,
    IvfConfigWithBase, QuantizerConfig, QuantizerType, SpannConfigWithBase,
};

#[derive(clap::ValueEnum, Clone, Debug, PartialEq)]
enum IndexTypeArgs {
    Hnsw,
    Ivf,
    Spann,
}

#[derive(clap::ValueEnum, Clone, Debug, PartialEq)]
enum QuantizerTypeArgs {
    ProductQuantizer,
    NoQuantizer,
}

#[derive(clap::ValueEnum, Clone, Debug, PartialEq)]
enum IntSeqEncodingTypeArgs {
    EliasFano,
    PlainEncoding,
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(long, default_value_t = false, required = false)]
    reindex: bool,

    #[arg(long = "quantizer-type", default_value_t = QuantizerTypeArgs::NoQuantizer, required = true, value_enum)]
    quantizer_type: QuantizerTypeArgs,

    #[arg(long = "int-seq-encoding-type", default_value_t = IntSeqEncodingTypeArgs::PlainEncoding, required = false, value_enum)]
    int_seq_encoding_type: IntSeqEncodingTypeArgs,

    #[arg(long = "index-type", default_value_t = IndexTypeArgs::Spann, required = false, value_enum)]
    index_type: IndexTypeArgs,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            reindex: false,
            quantizer_type: QuantizerTypeArgs::NoQuantizer,
            int_seq_encoding_type: IntSeqEncodingTypeArgs::PlainEncoding,
            index_type: IndexTypeArgs::Spann,
        }
    }
}

fn generate_config(args: &Args) -> IndexWriterConfig {
    let output_path = format!(
        "/tmp/{:?}_{}_{:?}_{:?}",
        args.index_type,
        if args.reindex {
            "reindex"
        } else {
            "no_reindex"
        },
        args.quantizer_type,
        args.int_seq_encoding_type,
    )
    .to_lowercase();

    let mut base_config = BaseConfig::default();
    base_config.reindex = args.reindex;
    base_config.dimension = 128;
    base_config.output_path = output_path;
    base_config.max_memory_size = 1024 * 1024 * 1024; // 1 GB
    base_config.file_size = 1024 * 1024 * 1024; // 1 GB
    base_config.index_type = index_writer::config::IndexType::Spann;

    let mut quantizer_config = QuantizerConfig::default();
    if args.quantizer_type == QuantizerTypeArgs::ProductQuantizer {
        quantizer_config.quantizer_type = QuantizerType::ProductQuantizer;
        quantizer_config.subvector_dimension = 8;
        quantizer_config.num_bits = 8;
        quantizer_config.num_training_rows = 10000;
        quantizer_config.max_iteration = 1000;
        quantizer_config.batch_size = 4;
    }

    let mut ivf_config = IvfConfig::default();
    if args.int_seq_encoding_type == IntSeqEncodingTypeArgs::EliasFano {
        ivf_config.posting_list_encoding_type = IntSeqEncodingType::EliasFano;
    }
    ivf_config.num_clusters = 10;
    ivf_config.num_data_points = 100000;
    ivf_config.max_clusters_per_vector = 1;
    ivf_config.max_iteration = 1000;
    ivf_config.batch_size = 4;
    ivf_config.max_posting_list_size = 1000;

    let mut hnsw_config = HnswConfig::default();
    hnsw_config.num_layers = 4;
    hnsw_config.max_num_neighbors = 32;
    hnsw_config.ef_construction = 200;

    match args.index_type {
        IndexTypeArgs::Hnsw => IndexWriterConfig::Hnsw(HnswConfigWithBase {
            base_config,
            quantizer_config,
            hnsw_config,
        }),
        IndexTypeArgs::Ivf => IndexWriterConfig::Ivf(IvfConfigWithBase {
            base_config,
            quantizer_config,
            ivf_config,
        }),
        IndexTypeArgs::Spann => IndexWriterConfig::Spann(SpannConfigWithBase {
            base_config,
            quantizer_config,
            hnsw_config,
            ivf_config,
        }),
    }
}

fn write_config_to_yaml_file(config: &IndexWriterConfig, file_path: &str) -> Result<()> {
    let yaml = serde_yaml::to_string(config).unwrap();
    let mut file = std::fs::File::create(file_path)?;
    file.write_all(yaml.as_bytes())?;
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    let config = generate_config(&args);
    write_config_to_yaml_file(&config, "/tmp/index_writer_config.yaml")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_args_parsing() {
        // Test parsing with all required arguments
        let args = Args::try_parse_from(
            [
                "write_index_writer_config",
                "--index-type",
                "hnsw",
                "--quantizer-type",
                "no-quantizer",
            ]
            .iter(),
        )
        .unwrap();
        assert_eq!(args.index_type, IndexTypeArgs::Hnsw);
        assert_eq!(args.quantizer_type, QuantizerTypeArgs::NoQuantizer);
        assert!(!args.reindex);

        // Test parsing with reindex flag
        let args = Args::try_parse_from(
            [
                "write_index_writer_config",
                "--index-type",
                "hnsw",
                "--quantizer-type",
                "no-quantizer",
                "--reindex",
            ]
            .iter(),
        )
        .unwrap();
        assert_eq!(args.index_type, IndexTypeArgs::Hnsw);
        assert_eq!(args.quantizer_type, QuantizerTypeArgs::NoQuantizer);
        assert!(args.reindex);

        // Test parsing with invalid index type
        let result = Args::try_parse_from(
            [
                "write_index_writer_config",
                "--index-type",
                "Invalid",
                "--quantizer-type",
                "no-quantizer",
            ]
            .iter(),
        );
        assert!(result.is_err());

        // Test parsing with invalid quantizer type
        let result = Args::try_parse_from(
            [
                "write_index_writer_config",
                "--index-type",
                "hnsw",
                "--quantizer-type",
                "Invalid",
            ]
            .iter(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_config_generation() {
        let mut args = Args::default();
        args.index_type = IndexTypeArgs::Hnsw;
        args.quantizer_type = QuantizerTypeArgs::NoQuantizer;
        args.reindex = false;

        let config = generate_config(&args);
        match config {
            IndexWriterConfig::Hnsw(config) => {
                assert_eq!(config.base_config.reindex, false);
                assert_eq!(config.base_config.dimension, 128);
                assert_eq!(
                    config.base_config.output_path,
                    "/tmp/hnsw_no_reindex_noquantizer_plainencoding"
                );
                assert_eq!(config.base_config.max_memory_size, 1024 * 1024 * 1024);
                assert_eq!(config.base_config.file_size, 1024 * 1024 * 1024);
                assert_eq!(
                    config.base_config.index_type,
                    index_writer::config::IndexType::Spann
                );

                assert_eq!(
                    config.quantizer_config.quantizer_type,
                    QuantizerType::NoQuantizer
                );

                assert_eq!(config.hnsw_config.num_layers, 4);
                assert_eq!(config.hnsw_config.max_num_neighbors, 32);
                assert_eq!(config.hnsw_config.ef_construction, 200);
            }
            _ => panic!("Expected Hnsw config"),
        }

        // Test with ProductQuantizer
        args.quantizer_type = QuantizerTypeArgs::ProductQuantizer;
        let config = generate_config(&args);
        match config {
            IndexWriterConfig::Hnsw(config) => {
                assert_eq!(
                    config.quantizer_config.quantizer_type,
                    QuantizerType::ProductQuantizer
                );
                assert_eq!(config.quantizer_config.subvector_dimension, 8);
                assert_eq!(config.quantizer_config.num_bits, 8);
                assert_eq!(config.quantizer_config.num_training_rows, 10000);
                assert_eq!(config.quantizer_config.max_iteration, 1000);
                assert_eq!(config.quantizer_config.batch_size, 4);
            }
            _ => panic!("Expected Hnsw config"),
        }

        // Test with Ivf
        args.index_type = IndexTypeArgs::Ivf;
        let config = generate_config(&args);
        match config {
            IndexWriterConfig::Ivf(config) => {
                assert_eq!(config.base_config.reindex, false);
                assert_eq!(config.base_config.dimension, 128);
                assert_eq!(
                    config.base_config.output_path,
                    "/tmp/ivf_no_reindex_productquantizer_plainencoding"
                );
                assert_eq!(config.base_config.max_memory_size, 1024 * 1024 * 1024);
                assert_eq!(config.base_config.file_size, 1024 * 1024 * 1024);
                assert_eq!(
                    config.base_config.index_type,
                    index_writer::config::IndexType::Spann
                );

                assert_eq!(
                    config.quantizer_config.quantizer_type,
                    QuantizerType::ProductQuantizer
                );

                assert_eq!(config.ivf_config.num_clusters, 10);
                assert_eq!(config.ivf_config.num_data_points, 100000);
                assert_eq!(config.ivf_config.max_clusters_per_vector, 1);
                assert_eq!(config.ivf_config.max_iteration, 1000);
                assert_eq!(config.ivf_config.batch_size, 4);
                assert_eq!(config.ivf_config.max_posting_list_size, 1000);
            }
            _ => panic!("Expected Ivf config"),
        }

        // Test with Spann
        args.index_type = IndexTypeArgs::Spann;
        let config = generate_config(&args);
        match config {
            IndexWriterConfig::Spann(config) => {
                assert_eq!(config.base_config.reindex, false);
                assert_eq!(config.base_config.dimension, 128);
                assert_eq!(
                    config.base_config.output_path,
                    "/tmp/spann_no_reindex_productquantizer_plainencoding"
                );
                assert_eq!(config.base_config.max_memory_size, 1024 * 1024 * 1024);
                assert_eq!(config.base_config.file_size, 1024 * 1024 * 1024);
                assert_eq!(
                    config.base_config.index_type,
                    index_writer::config::IndexType::Spann
                );

                assert_eq!(
                    config.quantizer_config.quantizer_type,
                    QuantizerType::ProductQuantizer
                );

                assert_eq!(config.hnsw_config.num_layers, 4);
                assert_eq!(config.hnsw_config.max_num_neighbors, 32);
                assert_eq!(config.hnsw_config.ef_construction, 200);

                assert_eq!(config.ivf_config.num_clusters, 10);
                assert_eq!(config.ivf_config.num_data_points, 100000);
                assert_eq!(config.ivf_config.max_clusters_per_vector, 1);
                assert_eq!(config.ivf_config.max_iteration, 1000);
                assert_eq!(config.ivf_config.batch_size, 4);
                assert_eq!(config.ivf_config.max_posting_list_size, 1000);
            }
            _ => panic!("Expected Spann config"),
        }
    }

    #[test]
    fn test_yaml_generation() {
        let mut args = Args::default();
        args.index_type = IndexTypeArgs::Hnsw;
        args.quantizer_type = QuantizerTypeArgs::NoQuantizer;
        args.reindex = false;

        let config = generate_config(&args);
        write_config_to_yaml_file(&config, "/tmp/index_writer_config.yaml").unwrap();

        // Read back and verify
        let content = std::fs::read_to_string("/tmp/index_writer_config.yaml").unwrap();
        let yaml = serde_yaml::to_string(&config).unwrap();
        assert_eq!(content, yaml);

        // Clean up
        std::fs::remove_file("/tmp/index_writer_config.yaml").unwrap();
    }
}
