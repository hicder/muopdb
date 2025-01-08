use std::sync::RwLock;

use anyhow::{Ok, Result};
use dashmap::DashMap;

use crate::multi_spann::builder::MultiSpannBuilder;
use crate::multi_spann::writer::MultiSpannWriter;
use crate::spann::builder::{SpannBuilder, SpannBuilderConfig};

pub struct MutableSegment {
    spann_builder_per_user: DashMap<u64, RwLock<SpannBuilder>>,
    config: SpannBuilderConfig,

    // Prevent a mutable segment from being modified after it is built.
    finalized: bool,
}

impl MutableSegment {
    pub fn new(config: SpannBuilderConfig) -> Result<Self> {
        Ok(Self {
            spann_builder_per_user: DashMap::new(),
            config,
            finalized: false,
        })
    }

    pub fn insert(&mut self, doc_id: u64, data: &[f32]) -> Result<()> {
        if self.finalized {
            return Err(anyhow::anyhow!("Cannot insert into a finalized segment"));
        }

        self.insert_for_user(0, doc_id, data)
    }

    /// Insert a document for a user
    pub fn insert_for_user(&self, user_id: u64, doc_id: u64, data: &[f32]) -> Result<()> {
        if self.finalized {
            return Err(anyhow::anyhow!("Cannot insert into a finalized segment"));
        }

        let spann_builder = self
            .spann_builder_per_user
            .entry(user_id)
            .or_insert_with(|| RwLock::new(SpannBuilder::new(self.config.clone()).unwrap()));
        spann_builder.write().unwrap().add(doc_id, data)?;
        Ok(())
    }

    pub fn build(&mut self, base_directory: String, name: String) -> Result<()> {
        if self.finalized {
            return Err(anyhow::anyhow!("Cannot build a finalized segment"));
        }

        let segment_directory = format!("{}/{}", base_directory, name);
        std::fs::create_dir_all(&segment_directory)?;

        let mut multi_spann_builder = MultiSpannBuilder::new(self.config.clone())?;
        multi_spann_builder.build()?;

        let multi_spann_writer = MultiSpannWriter::new(segment_directory);
        multi_spann_writer.write(&mut multi_spann_builder)?;
        self.finalized = true;
        Ok(())
    }
}

unsafe impl Send for MutableSegment {}

unsafe impl Sync for MutableSegment {}
