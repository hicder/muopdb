mod admin_server;
mod collection_catalog;
mod collection_manager;
mod collection_provider;
mod http_server;
mod index_server;
use std::net::SocketAddr;
use std::sync::Arc;

use admin_server::AdminServerImpl;
use clap::Parser;
use collection_catalog::CollectionCatalog;
use collection_manager::CollectionManager;
use collection_provider::CollectionProvider;
use http_server::HttpServer;
use index_server::IndexServerImpl;
use log::{debug, error, info};
use proto::admin::index_server_admin_server::IndexServerAdminServer;
use proto::muopdb::index_server_server::IndexServerServer;
use proto::muopdb::FILE_DESCRIPTOR_SET;
use tokio::spawn;
use tokio::sync::RwLock;
use tokio::time::sleep;
use tonic::transport::Server;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value_t = 9002)]
    port: u32,

    #[arg(long, default_value_t = 9003)]
    http_port: u16,

    #[arg(short, long)]
    node_id: u32,

    #[arg(long)]
    index_config_path: String,

    #[arg(long)]
    index_data_path: String,

    #[arg(long, default_value_t = 10)]
    num_ingestion_workers: u32,

    #[arg(long, default_value_t = 10)]
    num_flush_workers: u32,

    #[arg(long, default_value_t = true)]
    enable_auto_optimizing: bool,

    #[arg(long, default_value_t = 10000)]
    auto_optimizing_sleep_interval_ms: u64,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let arg = Args::parse();
    let addr: SocketAddr = format!("0.0.0.0:{}", arg.port).parse()?;
    let collection_config_path = arg.index_config_path;
    let collection_data_path = arg.index_data_path;
    let node_id = arg.node_id;

    info!("Node: {}, listening on port {}", node_id, arg.port);
    info!("Number of ingestion workers: {}", arg.num_ingestion_workers);
    info!("Number of flush workers: {}", arg.num_flush_workers);

    let collection_catalog = CollectionCatalog::new();
    let collection_provider = CollectionProvider::new(collection_data_path);
    let collection_manager = Arc::new(RwLock::new(CollectionManager::new(
        collection_config_path,
        collection_provider,
        collection_catalog,
        arg.num_ingestion_workers,
        arg.num_flush_workers,
    )));

    let collection_manager_clone = collection_manager.clone();
    let collection_manager_thread = spawn(async move {
        loop {
            if let Err(e) = collection_manager_clone
                .write()
                .await
                .check_for_update()
                .await
            {
                error!("Error checking for index manager update: {e}");
            }
            sleep(std::time::Duration::from_secs(60)).await;
        }
    });

    let collection_manager_clone_for_cleanup = collection_manager.clone();
    let automatic_segments_cleanup_thread = spawn(async move {
        if !arg.enable_auto_optimizing {
            info!("Automatic vacuum is disabled");
            return;
        }

        info!("Automatic vacuum is enabled");
        loop {
            collection_manager_clone_for_cleanup
                .read()
                .await
                .auto_optimize()
                .await
                .unwrap();
            sleep(std::time::Duration::from_secs(
                arg.auto_optimizing_sleep_interval_ms / 1000,
            ))
            .await;
        }
    });

    let mut ingestion_worker_threads = Vec::new();
    for i in 0..arg.num_ingestion_workers {
        let collection_manager_process_ops_clone = collection_manager.clone();
        let collection_manager_process_ops_thread = spawn(async move {
            loop {
                let processed_ops = collection_manager_process_ops_clone
                    .read()
                    .await
                    .process_ops(i)
                    .await
                    .unwrap();
                debug!("Processed {processed_ops} ops for worker {i}");
                // if there is no ops to process, sleep for 1 second
                if processed_ops == 0 {
                    sleep(std::time::Duration::from_secs(1)).await;
                }
            }
        });
        ingestion_worker_threads.push(collection_manager_process_ops_thread);
    }

    let mut flush_worker_threads = Vec::new();
    for i in 0..arg.num_flush_workers {
        let collection_manager_flush_clone = collection_manager.clone();
        let collection_manager_flush_thread = spawn(async move {
            loop {
                let flushed_ops = collection_manager_flush_clone
                    .read()
                    .await
                    .flush(i)
                    .await
                    .unwrap();
                debug!("Flushed {flushed_ops} ops for worker {i}");
                if flushed_ops == 0 {
                    sleep(std::time::Duration::from_secs(1)).await;
                }
            }
        });
        flush_worker_threads.push(collection_manager_flush_thread);
    }

    // Start the metrics server
    let http_server_addr = SocketAddr::new(addr.ip(), arg.http_port);
    info!("Starting HTTP server on {http_server_addr}");
    spawn(async move {
        if let Err(e) = HttpServer::new().serve(http_server_addr).await {
            error!("HTTP server error: {e}");
        }
    });

    let reflection_service = tonic_reflection::server::Builder::configure()
        .register_encoded_file_descriptor_set(FILE_DESCRIPTOR_SET)
        .build_v1()?;

    let reflection_service_v1_alpha = tonic_reflection::server::Builder::configure()
        .register_encoded_file_descriptor_set(FILE_DESCRIPTOR_SET)
        .build_v1alpha()?;

    let server_impl = IndexServerImpl::new(collection_manager.clone());
    let admin_impl = AdminServerImpl::new(collection_manager.clone());
    Server::builder()
        .add_service(IndexServerServer::new(server_impl))
        .add_service(IndexServerAdminServer::new(admin_impl))
        .add_service(reflection_service)
        .add_service(reflection_service_v1_alpha)
        .serve(addr)
        .await?;

    // TODO(hicder): Add graceful shutdown
    info!("Received signal, shutting down");
    collection_manager_thread.await?;
    automatic_segments_cleanup_thread.await?;
    for thread in ingestion_worker_threads {
        thread.await?;
    }
    for thread in flush_worker_threads {
        thread.await?;
    }
    Ok(())
}
