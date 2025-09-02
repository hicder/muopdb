use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::atomic::AtomicU64;

use anyhow::{Context, Result};
use config::enums::QuantizerType;
use index::collection::collection::Collection;
use index::collection::BoxedCollection;
use log::{debug, info, warn};
use quantization::noq::noq::NoQuantizerL2;
use quantization::pq::pq::ProductQuantizerL2;
use serde::{Deserialize, Serialize};
use utils::io::get_latest_version;

use crate::collection_catalog::CollectionCatalog;
use crate::collection_provider::CollectionProvider;

#[derive(Debug, Deserialize, Serialize)]
pub struct CollectionInfo {
    pub name: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct CollectionManagerConfig {
    pub collections: Vec<CollectionInfo>,
}

pub struct CollectionManager {
    config_path: String,
    collection_provider: CollectionProvider,
    collection_catalog: CollectionCatalog,
    latest_version: AtomicU64,
    num_ingestion_workers: u32,
    num_flush_workers: u32,
}

impl CollectionManager {
    pub fn new(
        config_path: String,
        collection_provider: CollectionProvider,
        collection_catalog: CollectionCatalog,
        num_ingestion_workers: u32,
        num_flush_workers: u32,
    ) -> Self {
        Self {
            config_path,
            collection_provider,
            collection_catalog,
            latest_version: AtomicU64::new(0),
            num_ingestion_workers,
            num_flush_workers,
        }
    }

    pub async fn collection_exists(&self, collection_name: &str) -> bool {
        self.collection_catalog
            .collection_exists(collection_name)
            .await
    }

    pub async fn get_collection(&self, collection_name: &str) -> Option<BoxedCollection> {
        self.collection_catalog
            .get_collection(collection_name)
            .await
    }

    pub async fn add_collection(
        &mut self,
        collection_name: String,
        collection_config: config::collection::CollectionConfig,
    ) -> Result<()> {
        let quantizer_type = &collection_config.quantization_type;
        match quantizer_type {
            QuantizerType::NoQuantizer => {
                // Create new directory
                Collection::<NoQuantizerL2>::init_new_collection(
                    format!(
                        "{}/{}",
                        self.collection_provider.data_directory(),
                        collection_name
                    ),
                    &collection_config,
                )?;
            }
            QuantizerType::ProductQuantizer => {
                Collection::<ProductQuantizerL2>::init_new_collection(
                    format!(
                        "{}/{}",
                        self.collection_provider.data_directory(),
                        collection_name
                    ),
                    &collection_config,
                )?;
            }
        }

        match self.collection_provider.read_collection(&collection_name) {
            Some(collection) => {
                self.collection_catalog
                    .add_collection(collection_name.clone(), collection)
                    .await;
            }
            None => {
                return Err(anyhow::anyhow!("Failed to read collection"));
            }
        }
        // Increment the latest version
        self.latest_version
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Write the collection manager config as latest version
        let toc_path = format!(
            "{}/version_{}",
            self.config_path,
            self.latest_version
                .load(std::sync::atomic::Ordering::Relaxed)
        );
        let all_collection_names = self
            .collection_catalog
            .get_all_collection_names_sorted()
            .await;
        let toc = CollectionManagerConfig {
            collections: all_collection_names
                .iter()
                .map(|name| CollectionInfo { name: name.clone() })
                .collect(),
        };
        serde_json::to_writer_pretty(std::fs::File::create(toc_path)?, &toc)?;

        Ok(())
    }

    fn get_collections_to_add(
        current_collection_names: &[String],
        new_collection_names: &[String],
    ) -> Vec<String> {
        let mut collections_to_add = vec![];
        for new_collection_name in new_collection_names {
            if !current_collection_names.contains(new_collection_name) {
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
            if !new_collection_names.contains(current_collection_name) {
                collections_to_remove.push(current_collection_name.clone());
            }
        }
        collections_to_remove
    }

    pub async fn check_for_update(&mut self) -> Result<()> {
        let latest_version =
            get_latest_version(&self.config_path).context("Failed to get latest version")?;
        if latest_version
            > self
                .latest_version
                .load(std::sync::atomic::Ordering::Relaxed)
        {
            info!("New version available: {latest_version}");
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

            self.latest_version
                .store(latest_version, std::sync::atomic::Ordering::Relaxed);
            let current_collection_names = self
                .collection_catalog
                .get_all_collection_names_sorted()
                .await;

            // TODO(hicder): Remove collections that are not in the new config
            let collections_to_add =
                Self::get_collections_to_add(&current_collection_names, &new_collection_names);
            for collection_name in collections_to_add.iter() {
                info!("Fetching collection {collection_name}");
                let collection_opt = self.collection_provider.read_collection(collection_name);
                if let Some(collection) = collection_opt {
                    self.collection_catalog
                        .add_collection(collection_name.clone(), collection)
                        .await;
                } else {
                    warn!("Failed to fetch collection {collection_name}");
                }
            }
        } else {
            info!("No new version available");
        }
        Ok(())
    }

    pub async fn process_ops(&self, worker_id: u32) -> Result<usize> {
        let mut processed_ops = 0;
        let collections = self
            .collection_catalog
            .get_all_collection_names_sorted()
            .await;

        for collection_name in collections {
            if self.get_worker_id(&collection_name, self.num_ingestion_workers) == worker_id {
                let collection = self
                    .collection_catalog
                    .get_collection(&collection_name)
                    .await
                    .unwrap();
                if collection.use_wal() {
                    debug!("Processing ops for collection {collection_name}");
                    processed_ops += collection.process_one_op().await?;
                }
            }
        }

        Ok(processed_ops)
    }

    pub async fn flush(&self, worker_id: u32) -> Result<usize> {
        let mut flushed_ops = 0;
        let collections = self
            .collection_catalog
            .get_all_collection_names_sorted()
            .await;

        for collection_name in collections {
            if self.get_worker_id(&collection_name, self.num_flush_workers) == worker_id {
                let collection = self
                    .collection_catalog
                    .get_collection(&collection_name)
                    .await
                    .unwrap();
                if collection.should_auto_flush() {
                    debug!("Automatically flushing collection {collection_name}");
                    flushed_ops += (!collection.flush().unwrap().is_empty()) as usize;
                }
            }
        }
        Ok(flushed_ops)
    }

    pub fn get_worker_id(&self, collection_name: &str, num_workers: u32) -> u32 {
        let mut hasher = DefaultHasher::new();
        collection_name.hash(&mut hasher);
        let hash = hasher.finish();
        hash as u32 % num_workers
    }

    /// Iterates through all collections and calls the `auto_optimize` method on each collection.
    pub async fn auto_optimize(&self) -> Result<()> {
        let collections = self
            .collection_catalog
            .get_all_collection_names_sorted()
            .await;

        info!("Auto optimizing collections: {collections:?}");

        for collection_name in collections {
            let collection = self
                .collection_catalog
                .get_collection(&collection_name)
                .await
                .unwrap();

            collection.auto_optimize().unwrap();
        }
        Ok(())
    }
}
