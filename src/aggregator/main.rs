mod aggregator;

use crate::aggregator::AggregatorServerImpl;
use clap::Parser;
use log::info;
use muopdb::muopdb::aggregator_server::AggregatorServer;
use rand::Rng;
use tonic::transport::Server;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Name of the person to greet
    #[arg(short, long, default_value_t = 0)]
    port: u32,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let arg = Args::parse();
    let port = if arg.port != 0 {
        arg.port
    } else {
        let mut rng = rand::thread_rng();
        rng.gen_range(9000..10000)
    };

    let addr = format!("127.0.0.1:{}", port).parse().unwrap();
    let server_impl = AggregatorServerImpl {};

    info!("Listening on port {}", port);

    Server::builder()
        .add_service(AggregatorServer::new(server_impl))
        .serve(addr)
        .await?;
    Ok(())
}
