use std::sync::RwLock;

use anyhow::{anyhow, Result};
use config::collection::CollectionConfig;
use dashmap::mapref::entry::Entry;
use dashmap::DashMap;
use log::debug;

use crate::spann::builder::{SpannBuilder, SpannBuilderConfig};

pub struct MultiSpannBuilder {
    config: CollectionConfig,
    inner_builders: DashMap<u128, RwLock<SpannBuilder>>,
    base_directory: String,
}

impl MultiSpannBuilder {
    pub fn new(config: CollectionConfig, base_directory: String) -> Result<Self> {
        Ok(Self {
            config,
            inner_builders: DashMap::new(),
            base_directory,
        })
    }

    pub fn insert(&self, user_id: u128, doc_id: u128, data: &[f32]) -> Result<()> {
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
        spann_builder.write().unwrap().add(doc_id, data)?;
        Ok(())
    }

    pub fn invalidate(&self, user_id: u128, doc_id: u128) -> Result<bool> {
        let result = match self.inner_builders.entry(user_id) {
            Entry::Occupied(entry) => Ok(entry),
            Entry::Vacant(_) => Err(anyhow!("No entry exists for user_id")),
        };

        let invalidated = result?.get().write().unwrap().invalidate(doc_id);

        Ok(invalidated)
    }

    pub fn build(&self) -> Result<()> {
        for entry in self.inner_builders.iter() {
            debug!("Building segment for user {}", entry.key());
            entry.value().write().unwrap().build()?;
        }
        Ok(())
    }

    pub fn user_ids(&self) -> Vec<u128> {
        self.inner_builders
            .iter()
            .map(|entry| *entry.key())
            .collect()
    }

    pub fn base_directory(&self) -> &str {
        &self.base_directory
    }

    /// This function will remove the builder for the given user id.
    /// If the builder is not found, it will return None.
    pub fn take_builder_for_user(&self, user_id: u128) -> Option<SpannBuilder> {
        self.inner_builders
            .remove(&user_id)
            .map(|builder| builder.1.into_inner().unwrap())
            .or(None)
    }

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
            let builder = ref_multi.value().read().unwrap();
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

        // Trying to invalidate from an inexistent user should return error
        assert!(multi_builder.invalidate(user_id_2, doc_id_2).is_err());

        let builder_lock = multi_builder.inner_builders.get(&user_id_1).unwrap();
        let builder = builder_lock.read().unwrap();
        assert!(builder.ivf_builder.is_valid_doc_id(doc_id_1));
        assert!(!builder.ivf_builder.is_valid_doc_id(doc_id_2));
    }
}
