use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result};
use log::debug;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use utils::io::get_latest_version;

#[derive(Debug, Deserialize, Serialize)]
pub struct ShardManagerConfig {
    pub version: u64,
    pub indices_to_shards: HashMap<String, Vec<ShardIdNodeId>>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ShardIdNodeId {
    pub shard_id: u32,
    pub node_id: u32,
}

pub struct ShardManager {
    config_directory: String,
    config: RwLock<Arc<ShardManagerConfig>>,
}

impl ShardManager {
    pub fn new(config_directory: String) -> Self {
        Self {
            config_directory,
            config: RwLock::new(Arc::new(ShardManagerConfig {
                version: 0,
                indices_to_shards: HashMap::new(),
            })),
        }
    }

    pub async fn check_for_update(&self) -> Result<()> {
        let latest_version =
            get_latest_version(&self.config_directory).context("Failed to get latest version")?;
        if latest_version > self.config.read().await.version {
            self.load_version(latest_version).await?;
        } else {
            debug!("No new version available");
        }
        Ok(())
    }

    async fn load_version(&self, version: u64) -> Result<()> {
        let config_path = format!("{}/version_{}", self.config_directory, version);
        let config_str =
            std::fs::read_to_string(&config_path).context("Failed to read config file")?;
        let mut config: ShardManagerConfig =
            serde_json::from_str(&config_str).context("Failed to parse config file")?;
        config.version = version;
        let config_arc = Arc::new(config);
        *self.config.write().await = config_arc;
        Ok(())
    }

    pub async fn get_nodes_for_index(&self, index_name: &str) -> Option<Vec<ShardIdNodeId>> {
        let config = self.config.read().await;
        config.indices_to_shards.get(index_name).cloned()
    }
}

// Test
#[cfg(test)]
mod tests {
    use tempdir::TempDir;

    use super::*;

    #[tokio::test]
    async fn test_get_nodes_for_index() {
        let temp_dir = TempDir::new("test_shard_manager").expect("Failed to create temp directory");
        let config_path = temp_dir
            .path()
            .to_str()
            .expect("Failed to convert temp dir path to string")
            .to_string();

        let config_v1 = ShardManagerConfig {
            version: 1,
            indices_to_shards: HashMap::from([
                (
                    "index1".to_string(),
                    vec![ShardIdNodeId {
                        shard_id: 0,
                        node_id: 1,
                    }],
                ),
                (
                    "index2".to_string(),
                    vec![ShardIdNodeId {
                        shard_id: 0,
                        node_id: 2,
                    }],
                ),
            ]),
        };

        // Write config to file
        let config_path_v1 = format!("{}/version_1", config_path);
        let config_json = serde_json::to_string(&config_v1).expect("Failed to serialize config");
        std::fs::write(config_path_v1, config_json).expect("Failed to write config to file");

        let shard_manager = ShardManager::new(config_path.clone());
        shard_manager
            .check_for_update()
            .await
            .expect("Failed to check for update");

        let nodes = shard_manager
            .get_nodes_for_index("index1")
            .await
            .expect("No nodes found for index1");
        assert_eq!(nodes.len(), 1);

        let config_v2 = ShardManagerConfig {
            version: 2,
            indices_to_shards: HashMap::from([
                (
                    "index1".to_string(),
                    vec![ShardIdNodeId {
                        shard_id: 0,
                        node_id: 1,
                    }],
                ),
                (
                    "index2".to_string(),
                    vec![ShardIdNodeId {
                        shard_id: 0,
                        node_id: 2,
                    }],
                ),
                (
                    "index3".to_string(),
                    vec![ShardIdNodeId {
                        shard_id: 0,
                        node_id: 3,
                    }],
                ),
            ]),
        };

        // Write config to file
        let config_path_v2 = format!("{}/version_2", config_path);
        let config_json = serde_json::to_string(&config_v2).expect("Failed to serialize config");
        std::fs::write(config_path_v2, config_json).expect("Failed to write config to file");

        shard_manager
            .check_for_update()
            .await
            .expect("Failed to check for update");

        let nodes = shard_manager
            .get_nodes_for_index("index1")
            .await
            .expect("No nodes found for index1");
        assert_eq!(nodes.len(), 1);
    }
}
