use clap::Parser;
use log::info;
use proto::muopdb::aggregator_client::AggregatorClient;
use proto::muopdb::GetRequest;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Name of the person to greet
    #[arg(short, long, default_value_t = 9001)]
    port: u32,
}

#[tokio::main]
async fn main() {
    env_logger::init();

    let arg = Args::parse();
    let addr = format!("http://127.0.0.1:{}", arg.port);
    let mut client = AggregatorClient::connect(addr).await.unwrap();

    let request = tonic::Request::new(GetRequest {
        key: "test".to_string(),
    });

    let response = client.get(request).await.unwrap();
    info!("Response: {:?}", response);
}
