use std::sync::Arc;

use log::{info, warn};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use utils::io::get_latest_version;

use crate::index_catalog::IndexCatalog;
use crate::index_provider::IndexProvider;
use anyhow::Context;
use anyhow::Result;

#[derive(Debug, Deserialize, Serialize)]
pub struct IndexConfig {
    pub name: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct IndexManagerConfig {
    pub indices: Vec<IndexConfig>,
}

pub struct IndexManager {
    config_path: String,
    index_provider: IndexProvider,
    index_catalog: Arc<Mutex<IndexCatalog>>,
    latest_version: u64,
}

impl IndexManager {
    pub fn new(
        config_path: String,
        index_provider: IndexProvider,
        index_catalog: Arc<Mutex<IndexCatalog>>,
    ) -> Self {
        Self {
            config_path,
            index_provider,
            index_catalog,
            latest_version: 0,
        }
    }

    fn get_indexes_to_add(
        current_index_names: &[String],
        new_index_names: &[String],
    ) -> Vec<String> {
        let mut indexes_to_add = vec![];
        for new_index_name in new_index_names {
            if !current_index_names.contains(&new_index_name) {
                indexes_to_add.push(new_index_name.clone());
            }
        }
        indexes_to_add
    }

    #[allow(dead_code)]
    fn get_indexes_to_remove(
        current_index_names: &[String],
        new_index_names: &[String],
    ) -> Vec<String> {
        let mut indexes_to_remove = vec![];
        for current_index_name in current_index_names {
            if !new_index_names.contains(&current_index_name) {
                indexes_to_remove.push(current_index_name.clone());
            }
        }
        indexes_to_remove
    }

    pub async fn check_for_update(&mut self) -> Result<()> {
        let latest_version = get_latest_version(&self.config_path)
            .context("Failed to get latest version")?;
        if latest_version > self.latest_version {
            info!("New version available: {}", latest_version);
            let latest_config_path = format!("{}/version_{}", self.config_path, latest_version);

            let config_str = std::fs::read_to_string(&latest_config_path)
                .context("Failed to read config file")?;
            let config: IndexManagerConfig = serde_json::from_str(&config_str)
                .context("Failed to parse config file")?;

            let new_index_names = config
                .indices
                .iter()
                .map(|x| x.name.clone())
                .collect::<Vec<String>>();

            self.latest_version = latest_version;
            let current_index_names = self
                .index_catalog
                .lock()
                .await
                .get_all_index_names_sorted()
                .await;

            // TODO(hicder): Remove indexes that are not in the new config
            let indexes_to_add = Self::get_indexes_to_add(&current_index_names, &new_index_names);
            for index_name in indexes_to_add.iter() {
                info!("Fetching index {}", index_name);
                let index = self.index_provider.read_index(index_name);
                if let Some(index) = index {
                    let idx = Arc::new(index);
                    self.index_catalog
                        .lock()
                        .await
                        .add_index(index_name.clone(), idx)
                        .await;
                } else {
                    warn!("Failed to fetch index {}", index_name);
                }
            }
        } else {
            info!("No new version available");
        }
        Ok(())
    }
}
