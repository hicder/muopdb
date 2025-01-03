use anyhow::{Ok, Result};

use crate::spann::builder::{SpannBuilder, SpannBuilderConfig};
use crate::spann::writer::SpannWriter;

pub struct MutableSegment {
    spann_builder: SpannBuilder,

    // Prevent a mutable segment from being modified after it is built.
    finalized: bool,
}

impl MutableSegment {
    pub fn new(config: SpannBuilderConfig) -> Result<Self> {
        let spann_builder = SpannBuilder::new(config)?;
        Ok(Self {
            spann_builder,
            finalized: false,
        })
    }

    pub fn insert(&mut self, doc_id: u64, data: &[f32]) -> Result<()> {
        if self.finalized {
            return Err(anyhow::anyhow!("Cannot insert into a finalized segment"));
        }

        self.spann_builder.add(doc_id, data)
    }

    pub fn build(&mut self, base_directory: String, name: String) -> Result<()> {
        if self.finalized {
            return Err(anyhow::anyhow!("Cannot build a finalized segment"));
        }

        let segment_directory = format!("{}/{}", base_directory, name);
        std::fs::create_dir_all(&segment_directory)?;

        self.spann_builder.build()?;
        let spann_writer = SpannWriter::new(segment_directory);
        spann_writer.write(&mut self.spann_builder)?;
        self.finalized = true;
        Ok(())
    }
}

unsafe impl Send for MutableSegment {}

unsafe impl Sync for MutableSegment {}
