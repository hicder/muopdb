mod aggregator;
mod node_manager;
mod shard_manager;

use std::sync::Arc;

use clap::Parser;
use node_manager::NodeManager;
use proto::aggregator::aggregator_server::AggregatorServer;
use shard_manager::ShardManager;
use tokio::sync::RwLock;
use tokio::time::sleep;
use tonic::transport::Server;
use tracing::{error, info};
use utils::tracing::{init_tracing, shutdown_tracing, TracingConfig};

use crate::aggregator::AggregatorServerImpl;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value_t = 9001)]
    port: u32,

    #[arg(long)]
    shard_manager_config_directory: String,

    #[arg(long)]
    node_manager_config_directory: String,

    #[arg(long, default_value_t = false, help = "Enable distributed tracing")]
    tracing_enabled: bool,

    #[arg(
        long,
        default_value = "http://localhost:4317",
        help = "OTLP endpoint for tracing"
    )]
    otlp_endpoint: String,

    #[arg(long, default_value_t = 1.0, help = "Tracing sampling rate (0.0-1.0)")]
    tracing_sampling_rate: f64,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arg = Args::parse();

    if arg.tracing_enabled {
        init_tracing(&TracingConfig {
            service_name: "muopdb-aggregator".to_string(),
            otlp_endpoint: arg.otlp_endpoint.clone(),
            sampling_rate: arg.tracing_sampling_rate,
            enabled: true,
        })?;
    } else {
        env_logger::init();
    }

    let addr = format!("0.0.0.0:{}", arg.port)
        .parse()
        .map_err(|e| format!("Failed to parse address: {}", e))?;
    let shard_manager_config_directory = arg.shard_manager_config_directory;
    let node_manager_config_directory = arg.node_manager_config_directory;

    info!("Listening on port {}", arg.port);

    let shard_manager = Arc::new(RwLock::new(ShardManager::new(
        shard_manager_config_directory,
    )));
    let shard_manager_for_update_thread = shard_manager.clone();
    let shard_manager_for_handler = shard_manager.clone();

    let shard_manager_thread = tokio::task::spawn(async move {
        loop {
            if let Err(e) = shard_manager_for_update_thread
                .read()
                .await
                .check_for_update()
                .await
            {
                error!("Error checking for node manager update: {}", e);
            }
            sleep(std::time::Duration::from_secs(10)).await;
        }
    });

    let node_manager = Arc::new(RwLock::new(NodeManager::new(node_manager_config_directory)));
    let node_manager_for_update_thread = node_manager.clone();
    let node_manager_for_handler = node_manager.clone();
    let node_manager_thread = tokio::task::spawn(async move {
        loop {
            if let Err(e) = node_manager_for_update_thread
                .read()
                .await
                .check_for_update()
                .await
            {
                error!("Error checking for node manager update: {}", e);
            }
            sleep(std::time::Duration::from_secs(10)).await;
        }
    });

    let server_impl =
        AggregatorServerImpl::new(shard_manager_for_handler, node_manager_for_handler);

    Server::builder()
        .add_service(AggregatorServer::new(server_impl))
        .serve(addr)
        .await?;

    shard_manager_thread
        .await
        .map_err(|e| format!("Shard manager thread error: {}", e))?;
    node_manager_thread
        .await
        .map_err(|e| format!("Node manager thread error: {}", e))?;

    if arg.tracing_enabled {
        shutdown_tracing();
    }
    Ok(())
}
