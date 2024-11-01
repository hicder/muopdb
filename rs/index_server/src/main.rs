mod index_catalog;
mod index_manager;
mod index_provider;
mod index_server;

use std::net::SocketAddr;
use std::sync::Arc;

use clap::Parser;
use index_catalog::IndexCatalog;
use index_manager::IndexManager;
use index_provider::IndexProvider;
use index_server::IndexServerImpl;
use log::info;
use proto::muopdb::index_server_server::IndexServerServer;
use tokio::spawn;
use tokio::sync::Mutex;
use tokio::time::sleep;
use tonic::transport::Server;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value_t = 9002)]
    port: u32,

    #[arg(short, long)]
    node_id: u32,

    #[arg(long)]
    index_config_path: String,

    #[arg(long)]
    index_data_path: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let arg = Args::parse();
    let addr: SocketAddr = format!("127.0.0.1:{}", arg.port).parse().unwrap();
    let index_config_path = arg.index_config_path;
    let index_data_path = arg.index_data_path;
    let node_id = arg.node_id;

    let index_catalog = Arc::new(Mutex::new(IndexCatalog::new()));
    let index_catalog_for_manager = index_catalog.clone();
    let index_catalog_for_server = index_catalog.clone();

    info!("Node: {}, listening on port {}", node_id, arg.port);

    let index_manager_thread = spawn(async move {
        let index_provider = IndexProvider::new(index_data_path);
        let mut index_manager =
            IndexManager::new(index_config_path, index_provider, index_catalog_for_manager);
        loop {
            index_manager.check_for_update().await;
            sleep(std::time::Duration::from_secs(60)).await;
        }
    });

    let server_impl = IndexServerImpl::new(index_catalog_for_server);
    Server::builder()
        .add_service(IndexServerServer::new(server_impl))
        .serve(addr)
        .await?;

    // TODO(hicder): Add graceful shutdown
    info!("Received signal, shutting down");
    index_manager_thread.await?;
    Ok(())
}
