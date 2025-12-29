use std::marker::PhantomData;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use num_traits::ops::bytes::ToBytes;
use utils::file_io::env::{Env, OpenResult};
use utils::file_io::FileIO;

use crate::vector::{StorageContext, VectorStorageConfig};

/// An asynchronous, block-based storage implementation for fixed-size vectors.
///
/// This storage reads vectors directly from a file using an `Env` abstraction,
/// supporting efficient random access to large-scale vector data.
pub struct AsyncFixedFileVectorStorage<T: ToBytes + Clone + Send + Sync> {
    file_io: Arc<dyn FileIO + Send + Sync>,
    num_vectors: usize,
    num_features: usize,
    offset: usize,
    _marker: PhantomData<T>,
}

impl<T: ToBytes + Clone + Send + Sync> AsyncFixedFileVectorStorage<T> {
    /// Creates a new `AsyncFixedFileVectorStorage` by opening the specified vector file.
    ///
    /// # Arguments
    /// * `env` - The environment for file I/O.
    /// * `file_path` - The path to the binary vector file.
    /// * `num_features` - The number of dimensions/features in each vector.
    ///
    /// # Returns
    /// * `Result<Self>` - A new storage instance or an error if initialization fails.
    pub async fn new(env: Arc<Box<dyn Env>>, file_path: String, num_features: usize) -> Result<Self> {
        let OpenResult { file_io, .. } = env.open(&file_path).await.map_err(|e| anyhow!("Failed to open vector file: {}", e))?;
        let num_vectors_data = file_io.read(0, 8).await.map_err(|e| anyhow!("Failed to read num_vectors: {}", e))?;
        let num_vectors = u64::from_le_bytes(num_vectors_data.try_into().unwrap()) as usize;

        Ok(Self {
            file_io,
            num_vectors,
            num_features,
            offset: 0,
            _marker: PhantomData,
        })
    }

    /// Creates a new `AsyncFixedFileVectorStorage` starting at a specific file offset.
    ///
    /// # Arguments
    /// * `env` - The environment for file I/O.
    /// * `file_path` - The path to the binary vector file.
    /// * `num_features` - The number of dimensions/features in each vector.
    /// * `offset` - The byte offset where the vector data section begins.
    ///
    /// # Returns
    /// * `Result<Self>` - A new storage instance or an error if initialization fails.
    pub async fn new_with_offset(
        env: Arc<Box<dyn Env>>,
        file_path: String,
        num_features: usize,
        offset: usize,
    ) -> Result<Self> {
        let OpenResult { file_io, .. } = env.open(&file_path).await.map_err(|e| anyhow!("Failed to open vector file: {}", e))?;

        let num_vectors_data = file_io.read(offset as u64, 8).await.map_err(|e| anyhow!("Failed to read num_vectors: {}", e))?;
        let num_vectors = u64::from_le_bytes(num_vectors_data.try_into().unwrap()) as usize;

        Ok(Self {
            file_io,
            num_vectors,
            num_features,
            offset,
            _marker: PhantomData,
        })
    }

    /// Calculates the size of a single vector in bytes.
    ///
    /// # Arguments
    /// * `num_features` - The number of features in the vector.
    ///
    /// # Returns
    /// * `usize` - The total size in bytes.
    fn vector_size_in_bytes(num_features: usize) -> usize {
        num_features * std::mem::size_of::<T>()
    }

    /// Retrieves a single vector by its internal ID.
    ///
    /// # Arguments
    /// * `id` - The internal index of the vector to retrieve.
    /// * `_context` - A storage context for tracking cache stats (currently used for its mutable reference).
    ///
    /// # Returns
    /// * `Result<Vec<T>>` - The retrieved vector or an error if the ID is out of bounds or reading fails.
    pub async fn get(&self, id: u32, _context: &mut impl StorageContext) -> Result<Vec<T>> {
        if id as usize >= self.num_vectors {
            return Err(anyhow!("index out of bounds"));
        }
        let start = self.offset + 8 + (id as usize) * Self::vector_size_in_bytes(self.num_features);
        let length = Self::vector_size_in_bytes(self.num_features) as u64;

        let data = self.file_io.read(start as u64, length).await.map_err(|e| anyhow!("Failed to read vector: {}", e))?;

        let item_size = std::mem::size_of::<T>();
        let num_items = data.len() / item_size;
        let mut result = Vec::with_capacity(num_items);
        for i in 0..num_items {
            let start = i * item_size;
            let end = start + item_size;
            let item = unsafe { std::ptr::read_unaligned(data[start..end].as_ptr() as *const T) };
            result.push(item);
        }

        Ok(result)
    }

    /// Retrieves multiple vectors by their internal IDs.
    ///
    /// # Arguments
    /// * `ids` - A slice of internal indices to retrieve.
    /// * `context` - A storage context for tracking cache stats.
    ///
    /// # Returns
    /// * `Result<Vec<Vec<T>>>` - A list of retrieved vectors or an error if any read fails.
    pub async fn multi_get(
        &self,
        ids: &[u32],
        context: &mut impl StorageContext,
    ) -> Result<Vec<Vec<T>>> {
        let mut results = Vec::with_capacity(ids.len());
        for id in ids {
            results.push(self.get(*id, context).await?);
        }
        Ok(results)
    }

    /// Returns the total number of vectors stored in this handler.
    ///
    /// # Returns
    /// * `usize` - The total vector count.
    pub fn num_vectors(&self) -> usize {
        self.num_vectors
    }

    /// Returns the configuration associated with this vector storage.
    ///
    /// # Returns
    /// * `VectorStorageConfig` - The storage configuration.
    pub fn config(&self) -> VectorStorageConfig {
        VectorStorageConfig {
            memory_threshold: 0,
            file_size: 0,
            num_features: self.num_features,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::BufWriter;
    use std::sync::Arc;

    use tempdir::TempDir;
    use utils::file_io::env::{DefaultEnv, EnvConfig, FileType};

    use super::*;
    use crate::utils::SearchContext;
    use crate::vector::file::FileBackedAppendableVectorStorage;

    #[tokio::test]
    async fn test_async_fixed_file_vector_storage() {
        let temp_dir = TempDir::new("async_vector_storage_test").unwrap();
        let base_directory = temp_dir.path().to_str().unwrap().to_string();

        let mut appendable_storage =
            FileBackedAppendableVectorStorage::<u32>::new(base_directory.clone(), 4, 8192, 4);
        for i in 0..257 {
            appendable_storage.append(&[i, i, i, i]).unwrap();
        }

        let vectors_path = format!("{}/vector_storage", base_directory);
        let mut vectors_file = File::create(vectors_path.clone()).unwrap();
        let mut vectors_buffer_writer = BufWriter::new(&mut vectors_file);

        appendable_storage
            .write(&mut vectors_buffer_writer)
            .unwrap();

        let config = EnvConfig {
            file_type: FileType::CachedStandard,
            ..EnvConfig::default()
        };
        let env: Arc<Box<dyn Env>> = Arc::new(Box::new(DefaultEnv::new(config)));

        let storage = AsyncFixedFileVectorStorage::<u32>::new(env.clone(), vectors_path, 4)
            .await
            .unwrap();

        assert_eq!(storage.num_vectors(), 257);

        let mut context = SearchContext::new(true);
        let vec0 = storage.get(0, &mut context).await.unwrap();
        assert_eq!(vec0, vec![0, 0, 0, 0]);

        let vec256 = storage.get(256, &mut context).await.unwrap();
        assert_eq!(vec256, vec![256, 256, 256, 256]);
    }

    #[tokio::test]
    async fn test_async_fixed_file_vector_storage_multi_get() {
        let temp_dir = TempDir::new("async_vector_multi_get_test").unwrap();
        let base_directory = temp_dir.path().to_str().unwrap().to_string();

        let mut appendable_storage =
            FileBackedAppendableVectorStorage::<f32>::new(base_directory.clone(), 4, 1024, 4);
        for i in 0..10 {
            let value = i as f32;
            appendable_storage
                .append(&[value, value, value, value])
                .unwrap();
        }

        let vectors_path = format!("{}/vector_storage", base_directory);
        let mut vectors_file = File::create(vectors_path.clone()).unwrap();
        let mut vectors_buffer_writer = BufWriter::new(&mut vectors_file);

        appendable_storage
            .write(&mut vectors_buffer_writer)
            .unwrap();

        let config = EnvConfig {
            file_type: FileType::CachedStandard,
            ..EnvConfig::default()
        };
        let env: Arc<Box<dyn Env>> = Arc::new(Box::new(DefaultEnv::new(config)));

        let storage = AsyncFixedFileVectorStorage::<f32>::new(env.clone(), vectors_path, 4)
            .await
            .unwrap();

        let mut context = SearchContext::new(false);
        let ids = [0u32, 3, 5, 9];
        let results = storage.multi_get(&ids, &mut context).await.unwrap();

        assert_eq!(results.len(), 4);
        assert_eq!(results[0], vec![0.0, 0.0, 0.0, 0.0]);
        assert_eq!(results[1], vec![3.0, 3.0, 3.0, 3.0]);
        assert_eq!(results[2], vec![5.0, 5.0, 5.0, 5.0]);
        assert_eq!(results[3], vec![9.0, 9.0, 9.0, 9.0]);
    }

    #[tokio::test]
    async fn test_async_fixed_file_vector_storage_out_of_bounds() {
        let temp_dir = TempDir::new("async_vector_oob_test").unwrap();
        let base_directory = temp_dir.path().to_str().unwrap().to_string();

        let mut appendable_storage =
            FileBackedAppendableVectorStorage::<u32>::new(base_directory.clone(), 4, 1024, 4);
        appendable_storage.append(&[1, 2, 3, 4]).unwrap();

        let vectors_path = format!("{}/vector_storage", base_directory);
        let mut vectors_file = File::create(vectors_path.clone()).unwrap();
        let mut vectors_buffer_writer = BufWriter::new(&mut vectors_file);

        appendable_storage
            .write(&mut vectors_buffer_writer)
            .unwrap();

        let config = EnvConfig {
            file_type: FileType::CachedStandard,
            ..EnvConfig::default()
        };
        let env: Arc<Box<dyn Env>> = Arc::new(Box::new(DefaultEnv::new(config)));

        let storage = AsyncFixedFileVectorStorage::<u32>::new(env.clone(), vectors_path, 4)
            .await
            .unwrap();

        let mut context = SearchContext::new(false);
        let result = storage.get(10, &mut context).await;
        assert!(result.is_err());
    }
}
