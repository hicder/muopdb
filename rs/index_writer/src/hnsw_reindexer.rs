use clap::Parser;
use index::hnsw::builder::HnswBuilder;
use index::hnsw::reader::HnswReader;
use index::hnsw::writer::HnswWriter;
use index::vector::VectorStorageConfig;
use log::error;
use quantization::pq::ProductQuantizer;

#[derive(Parser)]
struct Args {
    #[arg(short, long)]
    input_path: String,

    #[arg(short, long)]
    output_path: String,
}

pub fn main() {
    env_logger::init();

    let args = Args::parse();
    let reader = HnswReader::new(args.input_path);
    let index = reader.read::<ProductQuantizer>().unwrap();

    let vector_storage_config = VectorStorageConfig {
        memory_threshold: 1024 * 1024 * 1024,
        file_size: 1024 * 1024 * 1024,
        num_features: 16,
    };

    let mut builder =
        HnswBuilder::from_hnsw(index, args.output_path.clone(), vector_storage_config, 16);
    let writer = HnswWriter::new(args.output_path.clone());
    if let Err(e) = writer.write(&mut builder, true) {
        error!("Failed to write: {}", e);
        return;
    }

    println!("Done");
}
