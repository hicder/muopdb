use anyhow::Result;
use quantization::noq::noq::NoQuantizer;
use utils::distance::l2::L2DistanceCalculator;

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
        let posting_list_path = format!("{}/ivf", self.base_directory);
        let centroid_path = format!("{}/centroids", self.base_directory);

        let centroids = HnswReader::new(centroid_path).read::<NoQuantizer>()?;
        let posting_lists = IvfReader::new(posting_list_path).read::<NoQuantizer, L2DistanceCalculator>()?;

        Ok(Spann::new(centroids, posting_lists))
    }
}
