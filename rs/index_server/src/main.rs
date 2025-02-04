mod collection_catalog;
mod collection_manager;
mod collection_provider;
mod index_server;

use std::net::SocketAddr;
use std::sync::Arc;

use clap::Parser;
use collection_catalog::CollectionCatalog;
use collection_manager::CollectionManager;
use collection_provider::CollectionProvider;
use index_server::IndexServerImpl;
use log::{error, info};
use proto::muopdb::index_server_server::IndexServerServer;
use proto::muopdb::FILE_DESCRIPTOR_SET;
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
    let addr: SocketAddr = format!("0.0.0.0:{}", arg.port).parse()?;
    let collection_config_path = arg.index_config_path;
    let collection_data_path = arg.index_data_path;
    let node_id = arg.node_id;

    let collection_catalog = Arc::new(Mutex::new(CollectionCatalog::new()));
    let collection_catalog_for_manager = collection_catalog.clone();
    let collection_catalog_for_server = collection_catalog.clone();

    info!("Node: {}, listening on port {}", node_id, arg.port);

    let collection_provider = CollectionProvider::new(collection_data_path);
    let collection_manager = Arc::new(Mutex::new(CollectionManager::new(
        collection_config_path,
        collection_provider,
        collection_catalog_for_manager,
    )));

    let collection_manager_clone = collection_manager.clone();
    let collection_manager_thread = spawn(async move {
        loop {
            if let Err(e) = collection_manager_clone
                .lock()
                .await
                .check_for_update()
                .await
            {
                error!("Error checking for index manager update: {}", e);
            }
            sleep(std::time::Duration::from_secs(60)).await;
        }
    });

    let reflection_service = tonic_reflection::server::Builder::configure()
        .register_encoded_file_descriptor_set(FILE_DESCRIPTOR_SET)
        .build_v1()?;

    let reflection_service_v1_alpha = tonic_reflection::server::Builder::configure()
        .register_encoded_file_descriptor_set(FILE_DESCRIPTOR_SET)
        .build_v1alpha()?;

    let server_impl = IndexServerImpl::new(collection_catalog_for_server, collection_manager);
    Server::builder()
        .add_service(IndexServerServer::new(server_impl))
        .add_service(reflection_service)
        .add_service(reflection_service_v1_alpha)
        .serve(addr)
        .await?;

    // TODO(hicder): Add graceful shutdown
    info!("Received signal, shutting down");
    collection_manager_thread.await?;
    Ok(())
}
