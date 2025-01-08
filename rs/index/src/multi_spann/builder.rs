use std::sync::RwLock;

use anyhow::Result;
use dashmap::DashMap;

use crate::spann::builder::{SpannBuilder, SpannBuilderConfig};

pub struct MultiSpannBuilder {
    config: SpannBuilderConfig,
    inner_builders: DashMap<u64, RwLock<SpannBuilder>>,
}

impl MultiSpannBuilder {
    pub fn new(config: SpannBuilderConfig) -> Result<Self> {
        Ok(Self {
            config,
            inner_builders: DashMap::new(),
        })
    }

    pub fn insert(&self, user_id: u64, doc_id: u64, data: &[f32]) -> Result<()> {
        let spann_builder = self.inner_builders.entry(user_id).or_insert_with(|| {
            let mut config = self.config.clone();
            config.base_directory = format!("{}/{}", config.base_directory, user_id);
            RwLock::new(SpannBuilder::new(config).unwrap())
        });
        spann_builder.write().unwrap().add(doc_id, data)?;
        Ok(())
    }

    pub fn build(&self) -> Result<()> {
        for entry in self.inner_builders.iter() {
            entry.value().write().unwrap().build()?;
        }
        Ok(())
    }

    pub fn user_ids(&self) -> Vec<u64> {
        self.inner_builders
            .iter()
            .map(|entry| *entry.key())
            .collect()
    }

    pub fn base_directory(&self) -> &str {
        &self.config.base_directory
    }

    /// This function will remove the builder for the given user id.
    /// If the builder is not found, it will return None.
    pub fn take_builder_for_user(&self, user_id: u64) -> Option<SpannBuilder> {
        self.inner_builders
            .remove(&user_id)
            .map(|builder| builder.1.into_inner().unwrap())
            .or(None)
    }

    pub fn config(&self) -> &SpannBuilderConfig {
        &self.config
    }
}
