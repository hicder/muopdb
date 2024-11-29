use clap::{Parser, Subcommand};
use index_writer::config::{BaseConfig, HnswConfig, IndexWriterConfig, IvfConfig, QuantizerType};
use index_writer::index_writer::IndexWriter;
use index_writer::input::hdf5::Hdf5Reader;

#[derive(Subcommand, Debug)]
enum IndexTypeArgs {
    #[clap(name = "hnsw")]
    Hnsw(HnswArgs),

    #[clap(name = "ivf")]
    Ivf(IvfArgs),
}

#[derive(Parser, Debug, Default)]
struct HnswArgs {
    #[arg(long = "hnsw_num_layers", default_value_t = 8)]
    num_layers: u8,

    #[arg(long = "hnsw_max_num_neighbors", default_value_t = 16)]
    max_num_neighbors: usize,

    #[arg(long = "hnsw_ef_construction", default_value_t = 100)]
    ef_construction: u32,

    #[arg(long = "hnsw_subvector_dimension", default_value_t = 8)]
    subvector_dimension: usize,

    #[arg(long = "hnsw_num_bits", default_value_t = 8)]
    num_bits: u8,

    #[arg(long = "hnsw_num_training_rows", default_value_t = 1000)]
    num_training_rows: usize,

    #[arg(long = "hnsw_max_iteration", default_value_t = 5)]
    max_iteration: usize,

    #[arg(long = "hnsw_batch_size", default_value_t = 500)]
    batch_size: usize,

    #[arg(long = "hnsw_reindex", default_value_t = false)]
    reindex: bool,
}

#[derive(Parser, Debug, Default)]
struct IvfArgs {
    #[arg(long = "ivf_num_clusters", default_value_t = 100)]
    num_clusters: usize,

    #[arg(long = "ivf_num_data_points", default_value_t = 10000)]
    num_data_points: usize,

    #[arg(long = "ivf_max_clusters_per_vector", default_value_t = 4)]
    max_clusters_per_vector: usize,

    #[arg(long = "ivf_max_iteration", default_value_t = 5)]
    max_iteration: usize,

    #[arg(long = "ivf_batch_size", default_value_t = 500)]
    batch_size: usize,
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

    #[arg(long, default_value_t = 128)]
    dimension: usize,

    #[arg(long, default_value_t = 1 << 25)]
    vector_storage_memory_size: usize,

    #[arg(long, default_value_t = 1 << 30)]
    vector_storage_page_size: usize,

    #[command(subcommand)]
    index_type: IndexTypeArgs,
}

fn main() {
    env_logger::init();

    let arg = Args::parse();

    let base_config = BaseConfig {
        output_path: arg.output_path.clone(),
        dimension: arg.dimension,
        max_memory_size: arg.vector_storage_memory_size,
        file_size: arg.vector_storage_page_size,
    };
    let config = match arg.index_type {
        IndexTypeArgs::Hnsw(hnsw_args) => IndexWriterConfig::Hnsw(HnswConfig {
            base_config,

            num_layers: hnsw_args.num_layers,
            max_num_neighbors: hnsw_args.max_num_neighbors,
            ef_construction: hnsw_args.ef_construction,
            reindex: hnsw_args.reindex,

            quantizer_type: QuantizerType::ProductQuantizer,
            subvector_dimension: hnsw_args.subvector_dimension,
            num_bits: hnsw_args.num_bits,
            num_training_rows: hnsw_args.num_training_rows,

            max_iteration: hnsw_args.max_iteration,
            batch_size: hnsw_args.batch_size,
        }),
        IndexTypeArgs::Ivf(ivf_args) => IndexWriterConfig::Ivf(IvfConfig {
            base_config,

            num_clusters: ivf_args.num_clusters,
            num_data_points: ivf_args.num_data_points,
            max_clusters_per_vector: ivf_args.max_clusters_per_vector,

            max_iteration: ivf_args.max_iteration,
            batch_size: ivf_args.batch_size,
        }),
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
