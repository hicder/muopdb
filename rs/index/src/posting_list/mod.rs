use std::fs::File;
use std::io::BufWriter;
use std::mem::size_of;

use anyhow::Result;

pub mod combined_file;
pub mod file;
pub mod fixed_file;

/// Config for posting list storage.
pub struct PostingListStorageConfig {
    pub memory_threshold: usize,
    pub file_size: usize,
    pub num_clusters: usize,
}

pub struct PostingList<'a> {
    slices: Vec<&'a [u8]>,
}

pub struct PostingListIterator<'a> {
    slices: &'a Vec<&'a [u8]>,
    current_slice: usize,
    current_index: usize,
}

impl<'a> PostingList<'a> {
    pub fn new_with_slices(slices: Vec<&'a [u8]>) -> Self {
        PostingList { slices }
    }

    pub fn new() -> Self {
        PostingList::new_with_slices(Vec::new())
    }

    fn add_slice(&mut self, slice: &'a [u8]) {
        self.slices.push(slice);
    }

    pub fn iter(&'a self) -> PostingListIterator<'a> {
        PostingListIterator {
            slices: &self.slices,
            current_slice: 0,
            current_index: 0,
        }
    }
}

impl<'a> Iterator for PostingListIterator<'a> {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        while self.current_slice < self.slices.len() {
            let slice = self.slices[self.current_slice];
            if self.current_index < slice.len() / size_of::<u64>() {
                let idx = self.current_index * size_of::<u64>();
                let value =
                    u64::from_le_bytes(slice[idx..idx + size_of::<u64>()].try_into().unwrap());
                self.current_index += 1;
                return Some(value);
            }
            self.current_slice += 1;
            self.current_index = 0;
        }
        None
    }
}

/// Trait that defines the interface for posting list storage
/// This storage owns the actual vectors, and will return a reference to it
pub trait PostingListStorage<'a> {
    fn get(&'a self, id: u32) -> Result<PostingList<'a>>;

    fn append(&mut self, vector: &[u64]) -> Result<()>;

    // Number of posting lists in the storage
    fn len(&self) -> usize;

    // Return number of bytes written.
    fn write(&mut self, writer: &mut BufWriter<&mut File>) -> Result<usize>;

    // Return the config for this posting list storage. Useful when we want duplicate.
    fn config(&self) -> PostingListStorageConfig;
}
