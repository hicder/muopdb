use std::io::Read;
use std::path::Path;

use clap::Parser;
use index_writer::config::{
    HnswConfigWithBase, HnswIvfConfig, IndexWriterConfig, IvfConfigWithBase,
};
use index_writer::index_writer::IndexWriter;
use index_writer::input::hdf5::Hdf5Reader;

#[derive(clap::ValueEnum, Clone, Debug)]
enum IndexTypeArgs {
    Hnsw,
    Ivf,
    HnswIvf,
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

    #[arg(long, required = true)]
    output_path: String,

    #[arg(long, required = true)]
    config_path: String,

    #[arg(long = "index-type", required = true, value_enum)]
    index_type: IndexTypeArgs,
}

fn main() {
    env_logger::init();

    let arg = Args::parse();
    let mut file =
        std::fs::File::open(Path::new(&arg.config_path)).expect("Failed to open config file");
    let mut buf = String::new();
    file.read_to_string(&mut buf)
        .expect("Failed to read config file");

    let config = match arg.index_type {
        IndexTypeArgs::Hnsw => {
            let mut config: HnswConfigWithBase =
                serde_yaml::from_str(&buf).expect("Failed to parse config");
            config.base_config.output_path = arg.output_path;
            IndexWriterConfig::Hnsw(config)
        }
        IndexTypeArgs::Ivf => {
            let mut config: IvfConfigWithBase =
                serde_yaml::from_str(&buf).expect("Failed to parse config");
            config.base_config.output_path = arg.output_path;
            IndexWriterConfig::Ivf(config)
        }
        IndexTypeArgs::HnswIvf => {
            let mut config: HnswIvfConfig =
                serde_yaml::from_str(&buf).expect("Failed to parse config");
            config.base_config.output_path = arg.output_path;
            IndexWriterConfig::HnswIvf(config)
        }
    };

    // Open input
    let mut input = Hdf5Reader::new(1000, &arg.dataset_name, &arg.input_path)
        .expect("Failed to create Hdf5Reader");
    let mut index_writer = IndexWriter::new(config);

    // Process
    index_writer
        .process(&mut input)
        .expect("Index writer processing should succeed");
}
