use anyhow::Result;
use num_traits::ops::bytes::ToBytes;

pub mod file;
pub mod fixed_file;

/// Config for vector storage.
pub struct VectorStorageConfig {
    pub memory_threshold: usize,
    pub file_size: usize,
    pub num_features: usize,
}

pub trait StorageContext {
    fn should_record_pages(&self) -> bool;
    fn record_pages(&mut self, page_id: String);
}

pub enum VectorStorage<T: ToBytes + Clone> {
    FixedLocalFileBacked(fixed_file::FixedFileVectorStorage<T>),
}

impl<T: ToBytes + Clone> VectorStorage<T> {
    pub fn get_no_context(&self, id: u32) -> Result<&[T]> {
        match self {
            VectorStorage::FixedLocalFileBacked(storage) => storage.get_no_context(id),
        }
    }

    pub fn get(&self, id: u32, context: &mut impl StorageContext) -> Result<&[T]> {
        match self {
            VectorStorage::FixedLocalFileBacked(storage) => storage.get(id, context),
        }
    }

    pub fn multi_get(&self, ids: &[u32], context: &mut impl StorageContext) -> Result<Vec<&[T]>> {
        match self {
            VectorStorage::FixedLocalFileBacked(storage) => storage.multi_get(ids, context),
        }
    }

    pub fn num_vectors(&self) -> usize {
        match self {
            VectorStorage::FixedLocalFileBacked(storage) => storage.num_vectors(),
        }
    }

    pub fn config(&self) -> VectorStorageConfig {
        match self {
            VectorStorage::FixedLocalFileBacked(storage) => storage.config(),
        }
    }
}
