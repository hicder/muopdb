use std::fs::File;
use std::io::BufWriter;

use anyhow::Result;
use num_traits::ops::bytes::ToBytes;

pub mod file;
//pub mod fixed_file;

/// Config for inverted list storage.
pub struct InvertedListStorageConfig {
    pub memory_threshold: usize,
    pub file_size: usize,
    pub num_features: usize,
    pub num_clusters: usize,
}
