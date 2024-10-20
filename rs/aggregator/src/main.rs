mod aggregator;

use clap::Parser;
use log::info;
use proto::muopdb::aggregator_server::AggregatorServer;
use tonic::transport::Server;

use crate::aggregator::AggregatorServerImpl;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value_t = 9001)]
    port: u32,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let arg = Args::parse();

    let addr = format!("127.0.0.1:{}", arg.port).parse().unwrap();
    let server_impl = AggregatorServerImpl {};

    info!("Listening on port {}", arg.port);

    Server::builder()
        .add_service(AggregatorServer::new(server_impl))
        .serve(addr)
        .await?;
    Ok(())
}
