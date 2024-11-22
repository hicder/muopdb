use std::fs::File;
use std::io::BufWriter;

use anyhow::Result;

pub mod file;
//pub mod fixed_file;

/// Config for posting list storage.
pub struct PostingListStorageConfig {
    pub memory_threshold: usize,
    pub file_size: usize,
    pub num_clusters: usize,
}

/// Trait that defines the interface for posting list storage
/// This storage owns the actual vectors, and will return a reference to it
pub trait PostingListStorage {
    fn get(&self, id: u32) -> Result<Vec<usize>>;

    fn append(&mut self, vector: &[usize]) -> Result<()>;

    // Number of posting lists in the storage
    fn len(&self) -> usize;

    // Return number of bytes written.
    fn write(&mut self, writer: &mut BufWriter<&mut File>) -> Result<usize>;

    // Return the config for this posting list storage. Useful when we want duplicate.
    fn config(&self) -> PostingListStorageConfig;
}
