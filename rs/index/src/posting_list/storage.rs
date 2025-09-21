use anyhow::Result;

use super::combined_file::{FixedIndexFile, Header};

pub enum PostingListStorage {
    FixedLocalFile(FixedIndexFile),
}

impl PostingListStorage {
    pub fn new(base_directory: String) -> Self {
        Self::FixedLocalFile(FixedIndexFile::new(base_directory).unwrap())
    }

    pub fn get_posting_list(&self, id: usize) -> Result<&[u8]> {
        match self {
            PostingListStorage::FixedLocalFile(storage) => storage.get_posting_list(id),
        }
    }

    // Number of posting lists in the storage
    pub fn len(&self) -> usize {
        match self {
            PostingListStorage::FixedLocalFile(storage) => storage.header().num_clusters as usize,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get_centroid(&self, id: usize) -> Result<&[f32]> {
        match self {
            PostingListStorage::FixedLocalFile(storage) => storage.get_centroid(id),
        }
    }

    pub fn header(&self) -> &Header {
        match self {
            PostingListStorage::FixedLocalFile(storage) => storage.header(),
        }
    }

    pub fn get_doc_id(&self, id: usize) -> Result<u128> {
        match self {
            PostingListStorage::FixedLocalFile(storage) => storage.get_doc_id(id),
        }
    }
}
