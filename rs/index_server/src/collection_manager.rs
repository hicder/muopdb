use std::sync::Arc;

use anyhow::{Context, Result};
use log::{info, warn};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use utils::io::get_latest_version;

use crate::collection_catalog::CollectionCatalog;
use crate::collection_provider::CollectionProvider;

#[derive(Debug, Deserialize, Serialize)]
pub struct CollectionConfig {
    pub name: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct CollectionManagerConfig {
    pub collections: Vec<CollectionConfig>,
}

pub struct CollectionManager {
    config_path: String,
    collection_provider: CollectionProvider,
    collection_catalog: Arc<Mutex<CollectionCatalog>>,
    latest_version: u64,
}

impl CollectionManager {
    pub fn new(
        config_path: String,
        collection_provider: CollectionProvider,
        collection_catalog: Arc<Mutex<CollectionCatalog>>,
    ) -> Self {
        Self {
            config_path,
            collection_provider,
            collection_catalog,
            latest_version: 0,
        }
    }

    fn get_collections_to_add(
        current_collection_names: &[String],
        new_collection_names: &[String],
    ) -> Vec<String> {
        let mut collections_to_add = vec![];
        for new_collection_name in new_collection_names {
            if !current_collection_names.contains(&new_collection_name) {
                collections_to_add.push(new_collection_name.clone());
            }
        }
        collections_to_add
    }

    #[allow(dead_code)]
    fn get_collections_to_remove(
        current_collection_names: &[String],
        new_collection_names: &[String],
    ) -> Vec<String> {
        let mut collections_to_remove = vec![];
        for current_collection_name in current_collection_names {
            if !new_collection_names.contains(&current_collection_name) {
                collections_to_remove.push(current_collection_name.clone());
            }
        }
        collections_to_remove
    }

    pub async fn check_for_update(&mut self) -> Result<()> {
        let latest_version =
            get_latest_version(&self.config_path).context("Failed to get latest version")?;
        if latest_version > self.latest_version {
            info!("New version available: {}", latest_version);
            let latest_config_path = format!("{}/version_{}", self.config_path, latest_version);

            let config_str = std::fs::read_to_string(&latest_config_path)
                .context("Failed to read config file")?;
            let config: CollectionManagerConfig =
                serde_json::from_str(&config_str).context("Failed to parse config file")?;

            let new_collection_names = config
                .collections
                .iter()
                .map(|x| x.name.clone())
                .collect::<Vec<String>>();

            self.latest_version = latest_version;
            let current_collection_names = self
                .collection_catalog
                .lock()
                .await
                .get_all_collection_names_sorted()
                .await;

            // TODO(hicder): Remove collections that are not in the new config
            let collections_to_add =
                Self::get_collections_to_add(&current_collection_names, &new_collection_names);
            for collection_name in collections_to_add.iter() {
                info!("Fetching collection {}", collection_name);
                let collection_opt = self.collection_provider.read_collection(collection_name);
                if let Some(collection) = collection_opt {
                    self.collection_catalog
                        .lock()
                        .await
                        .add_collection(collection_name.clone(), collection)
                        .await;
                } else {
                    warn!("Failed to fetch collection {}", collection_name);
                }
            }
        } else {
            info!("No new version available");
        }
        Ok(())
    }
}
