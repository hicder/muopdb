use std::io::Read;
use std::path::Path;

use clap::Parser;
use index_writer::config::*;
use index_writer::index_writer::IndexWriter;
use index_writer::input::hdf5::Hdf5Reader;

#[derive(clap::ValueEnum, Clone, Debug)]
enum IndexTypeArgs {
    Hnsw,
    Ivf,
    Spann,
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
#[command(arg_required_else_help = true)]
struct Args {
    /// Input file
    #[arg(long, required = true)]
    input_path: String,

    #[arg(long, required = true)]
    dataset_name: String,

    #[arg(long, required = false)]
    output_path: Option<String>,

    #[arg(long, required = false)]
    config_path: Option<String>,

    #[arg(long, default_value_t = false, required = false)]
    reindex: bool,

    #[arg(long, default_value_t = false, required = false)]
    quantize: bool,

    #[arg(long = "index-type", required = true, value_enum)]
    index_type: IndexTypeArgs,
}

fn main() {
    env_logger::init();

    let arg = Args::parse();

    let config: IndexWriterConfig = match arg.config_path {
        Some(path) => {
            let mut file =
                std::fs::File::open(Path::new(&path)).expect("Failed to open config file");
            let mut buf = String::new();
            file.read_to_string(&mut buf)
                .expect("Failed to read config file");

            let output_path = arg
                .output_path
                .expect("output_path is required when providing a custom config file");
            match arg.index_type {
                IndexTypeArgs::Hnsw => {
                    let mut config: HnswConfigWithBase =
                        serde_yaml::from_str(&buf).expect("Failed to parse config");
                    config.base_config.output_path = output_path;
                    IndexWriterConfig::Hnsw(config)
                }
                IndexTypeArgs::Ivf => {
                    let mut config: IvfConfigWithBase =
                        serde_yaml::from_str(&buf).expect("Failed to parse config");
                    config.base_config.output_path = output_path;
                    IndexWriterConfig::Ivf(config)
                }
                IndexTypeArgs::Spann => {
                    let mut config: SpannConfigWithBase =
                        serde_yaml::from_str(&buf).expect("Failed to parse config");
                    config.base_config.output_path = output_path;
                    IndexWriterConfig::Spann(config)
                }
            }
        }
        None => {
            let quantizer_config = default_quantizer_config_for_bm(arg.quantize);
            let ivf_config = default_ivf_config_for_bm();
            let hnsw_config = default_hnsw_config_for_bm();
            match arg.index_type {
                IndexTypeArgs::Hnsw => IndexWriterConfig::Hnsw(HnswConfigWithBase {
                    base_config: default_base_config_for_bm(
                        arg.reindex,
                        arg.quantize,
                        IndexType::Hnsw,
                    ),
                    quantizer_config,
                    hnsw_config,
                }),
                IndexTypeArgs::Ivf => IndexWriterConfig::Ivf(IvfConfigWithBase {
                    base_config: default_base_config_for_bm(
                        arg.reindex,
                        arg.quantize,
                        IndexType::Ivf,
                    ),
                    quantizer_config,
                    ivf_config,
                }),
                IndexTypeArgs::Spann => IndexWriterConfig::Spann(SpannConfigWithBase {
                    base_config: default_base_config_for_bm(
                        arg.reindex,
                        arg.quantize,
                        IndexType::Spann,
                    ),
                    quantizer_config,
                    ivf_config,
                    hnsw_config,
                }),
            }
        }
    };

    // Open input
    let mut input = Hdf5Reader::new(1000, &arg.dataset_name, &arg.input_path)
        .expect("Failed to create Hdf5Reader");
    let mut index_writer = IndexWriter::new(config).expect("Failed to create index writer");

    // Process
    index_writer
        .process(&mut input)
        .expect("Index writer processing should succeed");
}
