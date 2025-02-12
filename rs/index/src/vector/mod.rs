use std::fs::File;
use std::io::BufWriter;

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

/// Trait that defines the interface for vector storage
/// This storage owns the actual vector, and will return a reference to it
// pub trait VectorStorage<T: ToBytes + Clone> {
//     fn get(&self, id: u32, context: &mut impl StorageContext) -> Result<&[T]>;

//     // Get multiple vectors at once. This is useful for batch operations,
//     // to avoid dynamic dispatch per point.
//     fn multi_get(&self, ids: &[u32], context: &mut impl StorageContext) -> Result<Vec<&[T]>>;

//     fn append(&mut self, vector: &[T]) -> Result<()>;

//     // Number of vectors in the storage
//     fn len(&self) -> usize;

//     // Return number of bytes written.
//     fn write(&self, writer: &mut BufWriter<&mut File>) -> Result<usize>;

//     // Return the config for this vector storage. Useful when we want duplicate.
//     fn config(&self) -> VectorStorageConfig;
// }

pub enum VectorStorage<T: ToBytes + Clone> {
    AppendableLocalFileBacked(file::FileBackedAppendableVectorStorage<T>),
    FixedLocalFileBacked(fixed_file::FixedFileVectorStorage<T>),
}

impl<T: ToBytes + Clone> VectorStorage<T> {
    pub fn get_no_context(&self, id: u32) -> Result<&[T]> {
        match self {
            VectorStorage::AppendableLocalFileBacked(storage) => storage.get_no_context(id),
            VectorStorage::FixedLocalFileBacked(storage) => storage.get_no_context(id),
        }
    }

    pub fn get(&self, id: u32, context: &mut impl StorageContext) -> Result<&[T]> {
        match self {
            VectorStorage::AppendableLocalFileBacked(storage) => storage.get(id, context),
            VectorStorage::FixedLocalFileBacked(storage) => storage.get(id, context),
        }
    }

    pub fn multi_get(&self, ids: &[u32], context: &mut impl StorageContext) -> Result<Vec<&[T]>> {
        match self {
            VectorStorage::AppendableLocalFileBacked(storage) => storage.multi_get(ids, context),
            VectorStorage::FixedLocalFileBacked(storage) => storage.multi_get(ids, context),
        }
    }

    pub fn append(&mut self, vector: &[T]) -> Result<()> {
        match self {
            VectorStorage::AppendableLocalFileBacked(storage) => storage.append(vector),
            VectorStorage::FixedLocalFileBacked(storage) => storage.append(vector),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            VectorStorage::AppendableLocalFileBacked(storage) => storage.len(),
            VectorStorage::FixedLocalFileBacked(storage) => storage.len(),
        }
    }

    pub fn write(&self, writer: &mut BufWriter<&mut File>) -> Result<usize> {
        match self {
            VectorStorage::AppendableLocalFileBacked(storage) => storage.write(writer),
            VectorStorage::FixedLocalFileBacked(storage) => storage.write(writer),
        }
    }

    pub fn config(&self) -> VectorStorageConfig {
        match self {
            VectorStorage::AppendableLocalFileBacked(storage) => storage.config(),
            VectorStorage::FixedLocalFileBacked(storage) => storage.config(),
        }
    }
}
