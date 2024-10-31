use clap::Parser;
use index_writer::config::IndexWriterConfig;
use index_writer::index_writer::IndexWriter;
use index_writer::input::hdf5::Hdf5Reader;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Input file
    #[arg(short, long)]
    input_path: String,

    #[arg(short, long)]
    dataset_name: String,

    #[arg(short, long)]
    output_path: String,

    #[arg(short, long, default_value_t = 8)]
    num_layers: u8,

    #[arg(short, long, default_value_t = 10)]
    max_num_neighbors: usize,

    #[arg(short, long, default_value_t = 10)]
    ef_construction: u32,

    #[arg(short, long, default_value_t = 128)]
    dimension: usize,

    #[arg(short, long, default_value_t = 8)]
    subvector_dimension: usize,

    #[arg(short, long, default_value_t = 8)]
    num_bits: u8,

    #[arg(short, long, default_value_t = 1000)]
    num_training_rows: usize,

    #[arg(short, long, default_value_t = 5)]
    max_iteration: usize,

    #[arg(short, long, default_value_t = 500)]
    batch_size: usize,

    #[arg(short, long, default_value_t = 1 << 25)]
    vector_storage_memory_size: usize,

    #[arg(short, long, default_value_t = 1 << 30)]
    vector_storage_page_size: usize,
}

fn main() {
    env_logger::init();

    let arg = Args::parse();
    let mut config = IndexWriterConfig::default();
    config.output_path = arg.output_path.clone();
    config.max_num_neighbors = arg.max_num_neighbors;
    config.num_layers = arg.num_layers;
    config.ef_construction = arg.ef_construction;
    config.dimension = arg.dimension;
    config.subvector_dimension = arg.subvector_dimension;
    config.num_bits = arg.num_bits;
    config.num_training_rows = arg.num_training_rows;
    config.max_iteration = arg.max_iteration;
    config.batch_size = arg.batch_size;
    config.max_memory_size = arg.vector_storage_memory_size;
    config.file_size = arg.vector_storage_page_size;

    // Open input
    let mut input = Hdf5Reader::new(1000, &arg.dataset_name, &arg.input_path).unwrap();
    let mut index_writer = IndexWriter::new(config);

    // Process
    index_writer.process(&mut input).unwrap();
}
