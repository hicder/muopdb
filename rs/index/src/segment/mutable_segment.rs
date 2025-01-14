use anyhow::{Ok, Result};
use config::collection::CollectionConfig;

use crate::multi_spann::builder::MultiSpannBuilder;
use crate::multi_spann::writer::MultiSpannWriter;

pub struct MutableSegment {
    multi_spann_builder: MultiSpannBuilder,

    // Prevent a mutable segment from being modified after it is built.
    finalized: bool,
}

impl MutableSegment {
    pub fn new(config: CollectionConfig, base_directory: String) -> Result<Self> {
        Ok(Self {
            multi_spann_builder: MultiSpannBuilder::new(config, base_directory)?,
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

        self.multi_spann_builder.insert(user_id, doc_id, data)?;
        Ok(())
    }

    pub fn build(&mut self, base_directory: String, name: String) -> Result<()> {
        if self.finalized {
            return Err(anyhow::anyhow!("Cannot build a finalized segment"));
        }

        let segment_directory = format!("{}/{}", base_directory, name);
        std::fs::create_dir_all(&segment_directory)?;

        self.multi_spann_builder.build()?;

        let multi_spann_writer = MultiSpannWriter::new(segment_directory);
        multi_spann_writer.write(&mut self.multi_spann_builder)?;
        self.finalized = true;
        Ok(())
    }
}

unsafe impl Send for MutableSegment {}

unsafe impl Sync for MutableSegment {}
