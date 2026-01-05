use clap::Parser;
use index::hnsw::reader::HnswReader;
use index::hnsw::utils::GraphTraversal;
use quantization::pq::pq::ProductQuantizer;
use utils::distance::l2::L2DistanceCalculator;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    // Index path
    #[arg(short, long)]
    index_path: String,

    #[arg(short, long)]
    layers: Option<Vec<u8>>,

    // Number of points per layer
    #[arg(long, default_value_t = 20)]
    points_per_layer: usize,

    #[arg(long, default_value_t = 100)]
    points_per_layer_0: usize,
}

use std::sync::Arc;

use utils::file_io::env::{DefaultEnv, Env, EnvConfig, FileType};

#[tokio::main]
pub async fn main() {
    env_logger::init();

    let arg = Args::parse();
    println!("Index path: {}", arg.index_path);

    let layers = arg.layers.unwrap_or(vec![]);
    let points_per_layer = arg.points_per_layer;
    let points_per_layer_0 = arg.points_per_layer_0;

    let env: Arc<Box<dyn Env>> = Arc::new(Box::new(DefaultEnv::new(EnvConfig {
        file_type: FileType::CachedStandard,
        ..EnvConfig::default()
    })));

    let reader = HnswReader::new(arg.index_path);
    let hnsw = reader
        .read::<ProductQuantizer<L2DistanceCalculator>>(env)
        .await
        .unwrap();

    let header = hnsw.get_header();
    println!("Header: {:?}", header);

    let entry_points = hnsw.get_entry_point_top_layer().await;
    println!("Entry points: {:?}", entry_points);

    for layer in layers {
        println!("======= Layer: {} =======", layer);
        let predicate = |l: u8, node_id: u32| -> bool {
            if l == 0 {
                node_id < points_per_layer_0 as u32
            } else {
                node_id < points_per_layer as u32
            }
        };
        hnsw.print_graph(layer, predicate);
    }
}
