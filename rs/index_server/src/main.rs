mod index_server;

use std::net::SocketAddr;

use clap::Parser;
use index_server::IndexServerImpl;
use log::info;
use proto::muopdb::index_server_server::IndexServerServer;
use tonic::transport::Server;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value_t = 9002)]
    port: u32,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let arg = Args::parse();
    let addr: SocketAddr = format!("127.0.0.1:{}", arg.port).parse().unwrap();
    let server_impl = IndexServerImpl {};

    info!("Listening on port {}", arg.port);

    Server::builder()
        .add_service(IndexServerServer::new(server_impl))
        .serve(addr)
        .await?;
    Ok(())
}
