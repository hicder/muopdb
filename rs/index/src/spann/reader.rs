use anyhow::Result;
use quantization::no_op::NoQuantizer;

use super::index::Spann;
use crate::hnsw::reader::HnswReader;
use crate::ivf::reader::IvfReader;

pub struct SpannReader {
    base_directory: String,
}

impl SpannReader {
    pub fn new(base_directory: String) -> Self {
        Self { base_directory }
    }

    pub fn read(&self) -> Result<Spann> {
        let centroids = HnswReader::new(self.base_directory.clone()).read::<NoQuantizer>()?;
        let posting_lists = IvfReader::new(self.base_directory.clone()).read()?;

        Ok(Spann::new(centroids, posting_lists))
    }
}
