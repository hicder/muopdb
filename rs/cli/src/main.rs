use clap::Parser;
use log::info;
use proto::muopdb::aggregator_client::AggregatorClient;
use proto::muopdb::index_server_client::IndexServerClient;
use proto::muopdb::{GetRequest, SearchRequest};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Name of the person to greet
    #[arg(short, long, default_value_t = 9001)]
    port: u32,

    #[arg(short, long, default_value_t = 0)]
    node_type: u32,
}

#[tokio::main]
async fn main() {
    env_logger::init();

    let arg = Args::parse();
    let addr = format!("http://127.0.0.1:{}", arg.port);

    let node_type = arg.node_type;
    info!("Node type: {}", node_type);

    if node_type == 0 {
        let mut client = AggregatorClient::connect(addr).await.unwrap();
        let request = tonic::Request::new(GetRequest {
            key: "test".to_string(),
        });

        let response = client.get(request).await.unwrap();
        info!("Response: {:?}", response);
    } else if node_type == 1 {
        let mut client = IndexServerClient::connect(addr).await.unwrap();
        let vec = (0..128).map(|_| rand::random::<f32>()).collect::<Vec<_>>();
        let request = tonic::Request::new(SearchRequest {
            index_name: "hieu-1".to_string(),
            vector: vec,
            top_k: 10,
        });
        let response = client.search(request).await.unwrap();
        info!("Response: {:?}", response);
    }
}
