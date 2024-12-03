use clap::Parser;
use index::hnsw::reader::HnswReader;
use index::hnsw::utils::GraphTraversal;
use quantization::pq::ProductQuantizer;

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

pub fn main() {
    env_logger::init();

    let arg = Args::parse();
    println!("Index path: {}", arg.index_path);

    let layers = arg.layers.unwrap_or(vec![]);
    let points_per_layer = arg.points_per_layer;
    let points_per_layer_0 = arg.points_per_layer_0;

    let reader = HnswReader::new(arg.index_path);
    let hnsw = reader.read::<ProductQuantizer>().unwrap();

    let header = hnsw.get_header();
    println!("Header: {:?}", header);

    let entry_points = hnsw.get_entry_point_top_layer();
    println!("Entry points: {:?}", entry_points);

    for layer in layers {
        println!("======= Layer: {} =======", layer);
        let predicate = |l: u8, node_id: u32| -> bool {
            if l == 0 {
                return node_id < points_per_layer_0 as u32;
            }
            return node_id < points_per_layer as u32;
        };
        hnsw.print_graph(layer as u8, predicate);
    }
}
