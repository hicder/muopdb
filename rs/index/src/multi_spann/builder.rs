use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

use anyhow::{anyhow, Result};
use config::collection::CollectionConfig;
use dashmap::mapref::entry::Entry;
use dashmap::DashMap;
use lock_api::RwLockUpgradableReadGuard;
use log::debug;
use parking_lot::RwLock;
use utils::bloom_filter::blocked_bloom_filter::BlockedBloomFilter;

use crate::spann::builder::{SpannBuilder, SpannBuilderConfig};

#[derive(Hash)]
pub struct BloomFilterKey {
    user_id: u128,
    doc_id: u128,
}

pub struct MultiSpannBuilder {
    config: CollectionConfig,
    inner_builders: DashMap<u128, RwLock<SpannBuilder>>,
    bloom_filter: OnceLock<BlockedBloomFilter>,
    doc_id_counts: AtomicU64,
    base_directory: String,
}

impl MultiSpannBuilder {
    /// Creates a new `MultiSpannBuilder` with the specified configuration and base directory.
    ///
    /// # Arguments
    /// * `config` - The overall collection configuration.
    /// * `base_directory` - The base directory where user-specific indices will be built.
    ///
    /// # Returns
    /// * `Result<Self>` - A new `MultiSpannBuilder` instance or an error.
    pub fn new(config: CollectionConfig, base_directory: String) -> Result<Self> {
        Ok(Self {
            config,
            inner_builders: DashMap::new(),
            bloom_filter: OnceLock::new(),
            doc_id_counts: AtomicU64::new(0),
            base_directory,
        })
    }

    /// Inserts data for a given user and document ID.
    ///
    /// # Arguments
    ///
    /// * `user_id` - The ID of the user.
    /// * `doc_id` - The ID of the document.
    /// * `data` - The data to insert.
    pub fn insert(&self, user_id: u128, doc_id: u128, data: &[f32]) -> Result<u32> {
        let spann_builder = self.inner_builders.entry(user_id).or_insert_with(|| {
            let user_directory = format!("{}/{}", self.base_directory, user_id);
            RwLock::new(
                SpannBuilder::new(SpannBuilderConfig::from_collection_config(
                    &self.config,
                    user_directory,
                ))
                .unwrap(),
            )
        });
        let point_id = spann_builder.write().add(doc_id, data)?;
        self.doc_id_counts.fetch_add(1, Ordering::Relaxed);
        Ok(point_id)
    }

    /// Invalidates a document ID for a given user.
    ///
    /// # Arguments
    ///
    /// * `user_id` - The ID of the user.
    /// * `doc_id` - The ID of the document to invalidate.
    ///
    /// # Returns
    ///
    /// * `Ok(true)` if the document ID was successfully invalidated.
    /// * `Ok(false)` if the user ID does not exist.
    pub fn invalidate(&self, user_id: u128, doc_id: u128) -> Result<bool> {
        // Check if the user_id exists in the inner_builders map
        if let Entry::Occupied(entry) = self.inner_builders.entry(user_id) {
            // If the entry exists, invalidate the doc_id and return the result
            let effectively_invalidated = entry.get().write().invalidate(doc_id);
            if effectively_invalidated {
                self.doc_id_counts.fetch_sub(1, Ordering::Relaxed);
            }
            Ok(effectively_invalidated)
        } else {
            // If the entry does not exist, return false instead of error
            Ok(false)
        }
    }

    /// Checks if a document ID is valid for a given user.
    ///
    /// # Arguments
    ///
    /// * `user_id` - The ID of the user.
    /// * `doc_id` - The ID of the document to check.
    ///
    /// # Returns
    ///
    /// * `true` if the document ID is valid for the user.
    /// * `false` otherwise.
    pub fn is_valid_doc_id(&self, user_id: u128, doc_id: u128) -> bool {
        let entry = self.inner_builders.entry(user_id).or_insert_with(|| {
            let user_directory = format!("{}/{}", self.base_directory, user_id);
            RwLock::new(
                SpannBuilder::new(SpannBuilderConfig::from_collection_config(
                    &self.config,
                    user_directory,
                ))
                .unwrap(),
            )
        });
        let spann_builder = entry.read();
        spann_builder.is_valid_doc_id(doc_id)
    }

    /// Builds the index segment for each user.
    ///
    /// Iterates through all internal builders, triggers the final build process for each
    /// `SpannBuilder`, and constructs a blocked bloom filter used for optimizing future deletions.
    ///
    /// # Returns
    /// * `Result<()>` - `Ok(())` if building all segments and the bloom filter succeeds, or an error.
    pub fn build(&self) -> Result<()> {
        let mut bloom_filter = BlockedBloomFilter::new(
            self.doc_id_counts.load(Ordering::Relaxed) as usize,
            self.config.fpr,
        );
        for entry in self.inner_builders.iter() {
            // Step 1: Build bloom filter
            let spann_builder = entry.value().upgradable_read();
            for doc_id in spann_builder.ivf_builder.doc_id_mapping() {
                let key = BloomFilterKey {
                    user_id: *entry.key(),
                    doc_id: *doc_id,
                };
                bloom_filter.insert::<BloomFilterKey>(&key);
            }
            // Step 2: Build segment
            debug!("Building segment for user {}", entry.key());
            let mut writable_spann_builder = RwLockUpgradableReadGuard::upgrade(spann_builder);
            writable_spann_builder.build()?;
        }

        match self.bloom_filter.set(bloom_filter) {
            Ok(_) => Ok(()),
            Err(_) => Err(anyhow!("build() was called multiple times")),
        }
    }

    /// Returns a list of all user IDs that have data in this builder.
    ///
    /// # Returns
    /// * `Vec<u128>` - A vector of 128-bit user IDs.
    pub fn user_ids(&self) -> Vec<u128> {
        self.inner_builders
            .iter()
            .map(|entry| *entry.key())
            .collect()
    }

    /// Returns the base directory of the builder.
    ///
    /// # Returns
    /// * `&str` - The base directory path string.
    pub fn base_directory(&self) -> &str {
        &self.base_directory
    }

    /// Removes and returns the `SpannBuilder` associated with a specific user ID.
    ///
    /// # Arguments
    /// * `user_id` - The ID of the user whose builder should be removed.
    ///
    /// # Returns
    /// * `Option<SpannBuilder>` - The user's `SpannBuilder` if it exists, otherwise `None`.
    pub fn take_builder_for_user(&self, user_id: u128) -> Option<SpannBuilder> {
        self.inner_builders
            .remove(&user_id)
            .map(|builder| builder.1.into_inner())
            .or(None)
    }

    /// Returns a reference to the built bloom filter, if it has been constructed.
    ///
    /// # Returns
    /// * `Option<&BlockedBloomFilter>` - A reference to the bloom filter or `None` if not yet built.
    pub fn bloom_filter(&self) -> Option<&BlockedBloomFilter> {
        self.bloom_filter.get()
    }

    /// Returns a reference to the collection configuration.
    ///
    /// # Returns
    /// * `&CollectionConfig` - A reference to the internal configuration.
    pub fn config(&self) -> &CollectionConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use config::collection::CollectionConfig;
    use tempdir::TempDir;

    use crate::multi_spann::builder::MultiSpannBuilder;

    #[test]
    fn test_multi_spann_builder() {
        let temp_dir = TempDir::new("test_multi_spann_builder").unwrap();
        let base_directory: String = temp_dir.path().to_str().unwrap().to_string();

        let spann_builder_config = CollectionConfig::default_test_config();
        let multi_builder = MultiSpannBuilder::new(spann_builder_config, base_directory.clone())
            .expect("Failed to create builder");

        let user_id_1 = 1u128;
        let user_id_2 = 2u128;
        let doc_id_1 = 101u128;
        let doc_id_2 = 102u128;
        let data_1 = [1.0, 2.0, 3.0, 4.0];
        let data_2 = [9.0, 10.0, 11.0, 12.0];
        assert!(multi_builder.insert(user_id_1, doc_id_1, &data_1).is_ok());
        assert!(multi_builder.insert(user_id_2, doc_id_2, &data_2).is_ok());

        let user_ids = multi_builder.user_ids();
        assert_eq!(user_ids.len(), 2);
        assert!(user_ids.contains(&user_id_1));
        assert!(user_ids.contains(&user_id_2));

        assert!(multi_builder.inner_builders.contains_key(&user_id_1));
        assert!(multi_builder.inner_builders.contains_key(&user_id_2));

        assert!(multi_builder.build().is_ok());

        // Verify the content of each builder
        for ref_multi in multi_builder.inner_builders.iter() {
            let user_id = *ref_multi.key();
            let builder = ref_multi.value().read();
            match user_id {
                1 => {
                    assert_eq!(builder.ivf_builder.vectors().num_vectors(), 1);
                    assert_eq!(
                        builder
                            .ivf_builder
                            .vectors()
                            .get_no_context(0)
                            .expect("Failed to read vector"),
                        &data_1
                    );
                    assert_eq!(builder.ivf_builder.posting_lists().len(), 1);
                    assert_eq!(
                        builder
                            .ivf_builder
                            .posting_lists()
                            .get(0)
                            .expect("Failed to read posting list")
                            .last()
                            .unwrap(),
                        0
                    );
                    assert_eq!(builder.ivf_builder.centroids().num_vectors(), 1);
                    assert_eq!(
                        builder
                            .ivf_builder
                            .centroids()
                            .get_no_context(0)
                            .expect("Failed to read centroid"),
                        &data_1
                    );
                }
                2 => {
                    assert_eq!(builder.ivf_builder.vectors().num_vectors(), 1);
                    assert_eq!(
                        builder
                            .ivf_builder
                            .vectors()
                            .get_no_context(0)
                            .expect("Failed to read vector"),
                        &data_2
                    );
                    assert_eq!(builder.ivf_builder.posting_lists().len(), 1);
                    assert_eq!(
                        builder
                            .ivf_builder
                            .posting_lists()
                            .get(0)
                            .expect("Failed to read posting list")
                            .last()
                            .unwrap(),
                        0
                    );
                    assert_eq!(builder.ivf_builder.centroids().num_vectors(), 1);
                    assert_eq!(
                        builder
                            .ivf_builder
                            .centroids()
                            .get_no_context(0)
                            .expect("Failed to read centroid"),
                        &data_2
                    );
                }
                _ => panic!("Unexpected user ID"),
            }
        }

        // Verify that the directories were created
        assert!(fs::metadata(format!("{}/{}", base_directory, user_id_1)).is_ok());
        assert!(fs::metadata(format!("{}/{}", base_directory, user_id_2)).is_ok());

        let builder_1 = multi_builder.take_builder_for_user(user_id_1);
        assert!(builder_1.is_some());

        let builder_2 = multi_builder.take_builder_for_user(user_id_2);
        assert!(builder_2.is_some());

        // Trying to take a non-existent builder should return None
        let non_existent_builder = multi_builder.take_builder_for_user(3u128);
        assert!(non_existent_builder.is_none());

        // The builders should be removed from multi_builder
        assert!(multi_builder.user_ids().is_empty());
    }

    #[test]
    fn test_multi_spann_builder_invalidate() {
        let temp_dir = TempDir::new("test_multi_spann_builder_invalidate").unwrap();
        let base_directory: String = temp_dir.path().to_str().unwrap().to_string();

        let spann_builder_config = CollectionConfig::default_test_config();
        let multi_builder = MultiSpannBuilder::new(spann_builder_config, base_directory.clone())
            .expect("Failed to create builder");

        let user_id_1 = 1u128;
        let user_id_2 = 2u128;
        let doc_id_1 = 101u128;
        let doc_id_2 = 102u128;
        let doc_id_3 = 103u128;
        let data_1 = [1.0, 2.0, 3.0, 4.0];
        let data_2 = [9.0, 10.0, 11.0, 12.0];
        assert!(multi_builder.insert(user_id_1, doc_id_1, &data_1).is_ok());
        assert!(multi_builder.insert(user_id_1, doc_id_2, &data_2).is_ok());

        assert!(multi_builder.invalidate(user_id_1, doc_id_2).is_ok());

        // Trying to invalidate a doc_id that is not in the builder should return false
        assert!(!multi_builder.invalidate(user_id_1, doc_id_3).unwrap());

        // Trying to invalidate from an inexistent user should return false
        assert!(!multi_builder.invalidate(user_id_2, doc_id_2).unwrap());

        let builder_lock = multi_builder.inner_builders.get(&user_id_1).unwrap();
        let builder = builder_lock.read();
        assert!(builder.ivf_builder.is_valid_doc_id(doc_id_1));
        assert!(!builder.ivf_builder.is_valid_doc_id(doc_id_2));
    }
}
