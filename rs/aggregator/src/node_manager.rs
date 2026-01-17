use std::collections::HashSet;
use std::sync::Arc;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use utils::io::get_latest_version;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeInfo {
    pub node_id: u32,
    pub ip: String,
    pub port: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct NodeManagerConfig {
    pub version: u64,
    pub nodes: Vec<NodeInfo>,
}

pub struct NodeManager {
    pub config_path: String,
    pub nodes: RwLock<Arc<NodeManagerConfig>>,
}

impl NodeManager {
    pub fn new(config_path: String) -> Self {
        Self {
            config_path,
            nodes: RwLock::new(Arc::new(NodeManagerConfig {
                version: 0,
                nodes: vec![],
            })),
        }
    }

    pub async fn get_nodes(&self, node_id: &HashSet<u32>) -> Vec<NodeInfo> {
        let mut ret: Vec<NodeInfo> = vec![];
        let nodes = self.nodes.read().await.clone();
        for node in nodes.nodes.iter() {
            if node_id.contains(&node.node_id) {
                ret.push(node.clone());
            }
        }
        ret
    }

    pub async fn check_for_update(&self) -> Result<()> {
        let latest_version = get_latest_version(&self.config_path)?;
        if latest_version > self.nodes.read().await.clone().version {
            self.load_version(latest_version).await?;
        }
        Ok(())
    }

    pub async fn load_version(&self, version: u64) -> Result<()> {
        let config_path = format!("{}/version_{}", self.config_path, version);
        let config_str = std::fs::read_to_string(&config_path)?;
        let mut config: NodeManagerConfig = serde_json::from_str(&config_str)?;
        config.version = version;
        let config_arc = Arc::new(config);
        *self.nodes.write().await = config_arc;
        Ok(())
    }
}

// Test
#[cfg(test)]
mod tests {
    use tempdir::TempDir;
    use tracing::error;

    use super::*;

    #[tokio::test]
    async fn test_get_nodes() {
        env_logger::init();
        let temp_dir = TempDir::new("test_node_manager").expect("Failed to create temp directory");
        let config_path = temp_dir
            .path()
            .to_str()
            .expect("Failed to convert temp dir path to string")
            .to_string();

        let config_v1 = NodeManagerConfig {
            version: 1,
            nodes: vec![NodeInfo {
                node_id: 1,
                ip: "127.0.0.1".to_string(),
                port: 9000,
            }],
        };

        // Write config to file
        let config_path_v1 = format!("{}/version_1", config_path);
        std::fs::write(
            config_path_v1,
            serde_json::to_string(&config_v1).expect("Failed to serialize config"),
        )
        .expect("Failed to write config to file");

        let node_manager = NodeManager::new(config_path.clone());
        if let Err(e) = node_manager.check_for_update().await {
            error!("Error checking for node manager update: {}", e);
        }
        let nodes = node_manager.get_nodes(&HashSet::from([1])).await;
        assert_eq!(nodes.len(), 1);

        let config_v2 = NodeManagerConfig {
            version: 2,
            nodes: vec![
                NodeInfo {
                    node_id: 1,
                    ip: "127.0.0.1".to_string(),
                    port: 9000,
                },
                NodeInfo {
                    node_id: 2,
                    ip: "127.0.0.1".to_string(),
                    port: 9001,
                },
            ],
        };

        // Write config to file
        let config_path_v2 = format!("{}/version_2", config_path);
        std::fs::write(
            config_path_v2,
            serde_json::to_string(&config_v2).expect("Failed to serialize config"),
        )
        .expect("Failed to write config to file");

        if let Err(e) = node_manager.check_for_update().await {
            error!("Error checking for node manager update: {}", e);
        }
        let nodes = node_manager.get_nodes(&HashSet::from([1, 2])).await;
        assert_eq!(nodes.len(), 2);
    }
}
