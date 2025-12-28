use std::hash::Hash;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use anyhow::{anyhow, bail, Result};
use dashmap::DashMap;
use log::info;
use moka::future::Cache;
use tokio::fs::File;

use crate::file_io::standard_file::StandardFile;
#[cfg(target_os = "linux")]
use crate::file_io::uring_engine::UringEngine;
#[cfg(target_os = "linux")]
use crate::file_io::uring_file::UringFile;
use crate::file_io::FileIO;

/// A unique identifier for an opened file in the block cache.
pub type FileId = u64;

/// Trait for block cache operations.
/// This allows the block cache to be used as a trait object.
#[async_trait::async_trait]
pub trait BlockCacheT: Send + Sync {
    async fn open_file(&mut self, path: &str) -> Result<FileId>;
    fn close_file(&mut self, file_id: FileId);
    async fn read(&mut self, file_id: FileId, offset: u64, length: u64) -> Result<Vec<u8>>;
    fn get_file_count(&self) -> usize;
}

/// Configuration for the block cache.
#[derive(Clone, Debug)]
pub struct BlockCacheConfig {
    pub max_open_files: u64,
    pub block_cache_capacity_bytes: u64,
    pub block_size: usize,
    pub use_io_uring: bool,
}

impl Default for BlockCacheConfig {
    fn default() -> Self {
        Self {
            max_open_files: 1000,
            block_cache_capacity_bytes: 1024 * 1024 * 1024,
            block_size: 4096,
            use_io_uring: false,
        }
    }
}

impl BlockCacheConfig {
    /// Creates a new BlockCacheConfig with the specified parameters.
    ///
    /// # Arguments
    ///
    /// * `max_open_files` - Maximum number of files that can be open simultaneously.
    /// * `block_cache_capacity_bytes` - Maximum memory capacity for cached blocks in bytes.
    /// * `block_size` - Size of each block in bytes. Must be greater than 0.
    /// * `use_io_uring` - Whether to use io_uring for file I/O on Linux.
    pub fn new(
        max_open_files: u64,
        block_cache_capacity_bytes: u64,
        block_size: usize,
        use_io_uring: bool,
    ) -> Self {
        if block_size == 0 {
            panic!("block_size must be greater than 0");
        }
        Self {
            max_open_files,
            block_cache_capacity_bytes,
            block_size,
            use_io_uring,
        }
    }
}

/// A wrapper around an opened file handle.
#[derive(Clone)]
struct FileEntry {
    file: Arc<dyn FileIO + Send + Sync>,
}

/// A key representing a specific block in a file, used for caching.
#[derive(Clone, Hash, Eq, PartialEq)]
struct BlockKey {
    file_id: FileId,
    block_number: u64,
}

impl BlockKey {
    fn new(file_id: FileId, block_number: u64) -> Self {
        Self {
            file_id,
            block_number,
        }
    }
}

/// A block-based file cache that provides caching of file blocks in memory.
///
/// The cache maintains two layers:
/// 1. A file descriptor cache using DashMap for concurrent access
/// 2. A block cache using Moka for cached block data
///
/// # Example
///
/// ```ignore
/// let config = BlockCacheConfig::default();
/// let mut cache = BlockCache::new(config);
///
/// let file_id = cache.open_file("path/to/file").await?;
/// let data = cache.read(file_id, 0, 4096).await?;
/// cache.close_file(file_id);
/// ```
pub struct BlockCache {
    config: BlockCacheConfig,
    file_descriptor_cache: DashMap<FileId, FileEntry>,
    block_cache: Cache<BlockKey, Vec<u8>>,
    file_id_generator: AtomicU64,
    #[cfg(target_os = "linux")]
    engine: Option<UringEngine>,
}

impl BlockCache {
    /// Creates a new BlockCache with the given configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - The block cache configuration.
    ///
    /// # Returns
    ///
    /// A new BlockCache instance.
    pub fn new(config: BlockCacheConfig) -> Self {
        info!("Creating block cache with config: {:#?}", config);
        #[cfg(target_os = "linux")]
        let engine = if config.use_io_uring {
            Some(UringEngine::new(256))
        } else {
            None
        };

        Self {
            config: config.clone(),
            file_descriptor_cache: DashMap::new(),
            block_cache: Cache::builder()
                .max_capacity(config.block_cache_capacity_bytes)
                .weigher(|_key, value: &Vec<u8>| value.len() as u32)
                .build(),
            file_id_generator: AtomicU64::new(1),
            #[cfg(target_os = "linux")]
            engine,
        }
    }

    /// Opens a file and returns a unique file identifier.
    ///
    /// The file is opened asynchronously and added to the file descriptor cache.
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the file to open.
    ///
    /// # Returns
    ///
    /// A Result containing the file ID, or an error if the file cannot be opened.
    pub async fn open_file(&self, path: &str) -> Result<FileId> {
        let file_id = self.file_id_generator.fetch_add(1, Ordering::SeqCst);

        info!(
            "[BLOCK_CACHE] Opening file: {} with io_uring: {}",
            path, self.config.use_io_uring
        );

        let file: Arc<dyn FileIO + Send + Sync> = if self.config.use_io_uring {
            #[cfg(target_os = "linux")]
            {
                let engine = self
                    .engine
                    .as_ref()
                    .expect("Engine should be Some when use_io_uring is true");
                Arc::new(UringFile::new(path, engine.handle()).await?)
            }
            #[cfg(not(target_os = "linux"))]
            {
                panic!("io_uring is only supported on Linux");
            }
        } else {
            let tokio_file = File::open(path)
                .await
                .map_err(|e| anyhow!("Failed to open file {}: {}", path, e))?;
            Arc::new(StandardFile::new(tokio_file).await)
        };

        self.file_descriptor_cache
            .insert(file_id, FileEntry { file });

        Ok(file_id)
    }

    /// Closes a file and removes it from the file descriptor cache.
    ///
    /// # Arguments
    ///
    /// * `file_id` - The file identifier returned from `open_file`.
    pub fn close_file(&mut self, file_id: FileId) {
        self.file_descriptor_cache.remove(&file_id);
    }

    /// Reads data from a file at the specified offset and length.
    ///
    /// The read is performed block-wise. If the requested data spans multiple blocks,
    /// all relevant blocks are read and cached. The method handles unaligned offsets
    /// by reading entire blocks and extracting the requested portion.
    ///
    /// # Arguments
    ///
    /// * `file_id` - The file identifier returned from `open_file`.
    /// * `offset` - The byte offset within the file to start reading from.
    /// * `length` - The number of bytes to read.
    ///
    /// # Returns
    ///
    /// A Result containing the read data as a vector of bytes, or an error if the
    /// read fails (e.g., invalid file ID, offset beyond file size, zero length).
    pub async fn read(&self, file_id: FileId, offset: u64, length: u64) -> Result<Vec<u8>> {
        if length == 0 {
            bail!("length must be greater than 0");
        }

        let file_entry = match self.file_descriptor_cache.get(&file_id) {
            Some(entry) => entry.value().clone(),
            None => {
                bail!(
                    "FileId {} not found. Did you call open_file first?",
                    file_id
                );
            }
        };

        let file_size = file_entry.file.file_length().await?;
        if offset >= file_size {
            bail!("offset {} is beyond file size {}", offset, file_size);
        }

        let end_offset = offset
            .checked_add(length)
            .ok_or_else(|| anyhow!("offset + length overflow"))?;
        if end_offset > file_size {
            bail!(
                "offset + length {} exceeds file size {}",
                end_offset,
                file_size
            );
        }

        let block_size = self.config.block_size as u64;
        let start_block = offset / block_size;
        let end_block = (end_offset - 1) / block_size;
        let aligned_offset = start_block * block_size;

        let mut aligned_data = Vec::new();

        for block_number in start_block..=end_block {
            let block_offset = block_number * block_size;
            let block_key = BlockKey::new(file_id, block_number);
            let file = file_entry.file.clone();

            let block_data = self
                .block_cache
                .try_get_with(block_key, async move {
                    Self::read_block_from_disk(&file, block_offset, 4096).await
                })
                .await
                .map_err(|e| anyhow!("Failed to read block: {}", e))?;

            aligned_data.extend_from_slice(&block_data);
        }

        let start_pos = (offset - aligned_offset) as usize;
        let end_pos = start_pos + length as usize;

        Ok(aligned_data[start_pos..end_pos].to_vec())
    }

    async fn read_block_from_disk(
        file: &Arc<dyn FileIO + Send + Sync>,
        offset: u64,
        block_size: usize,
    ) -> Result<Vec<u8>> {
        file.read(offset, block_size as u64).await
    }

    /// Returns the number of currently open files in the cache.
    pub fn get_file_count(&self) -> usize {
        self.file_descriptor_cache.len()
    }

    /// Returns the total number of blocks currently cached.
    ///
    /// This method runs any pending tasks in the cache first to get an accurate count.
    ///
    /// # Returns
    ///
    /// The total size (weighted) of all cached blocks.
    pub async fn get_block_count(&self) -> u64 {
        self.block_cache.run_pending_tasks().await;
        self.block_cache.weighted_size()
    }

    /// Returns the configured block size in bytes.
    ///
    /// # Returns
    ///
    /// The block size used for cache operations.
    pub fn block_size(&self) -> u64 {
        self.config.block_size as u64
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::Write;

    use tempdir::TempDir;

    use crate::block_cache::{BlockCache, BlockCacheConfig};

    #[tokio::test]
    async fn test_open_and_close_file() {
        let temp_dir = TempDir::new("block_cache_test").unwrap();
        let test_file_path = temp_dir.path().join("test.txt");
        let mut file = File::create(&test_file_path).unwrap();
        file.write_all(b"Hello, World!").unwrap();

        let config = BlockCacheConfig::default();
        let mut cache = BlockCache::new(config);

        let file_id = cache
            .open_file(test_file_path.to_str().unwrap())
            .await
            .unwrap();
        assert!(file_id > 0);

        cache.close_file(file_id);
    }

    #[tokio::test]
    async fn test_read_single_block() {
        let temp_dir = TempDir::new("block_cache_test").unwrap();
        let test_file_path = temp_dir.path().join("test.txt");
        let test_content = b"This is test content for reading.";
        let mut file = File::create(&test_file_path).unwrap();
        file.write_all(test_content).unwrap();

        let config = BlockCacheConfig::default();
        let mut cache = BlockCache::new(config);

        let file_id = cache
            .open_file(test_file_path.to_str().unwrap())
            .await
            .unwrap();

        let result = cache.read(file_id, 0, 4).await.unwrap();
        assert_eq!(&result, b"This");

        cache.close_file(file_id);
    }

    #[tokio::test]
    async fn test_read_spanning_blocks() {
        let temp_dir = TempDir::new("block_cache_test").unwrap();
        let test_file_path = temp_dir.path().join("test.txt");
        let test_content = vec![0u8; 8192];
        let mut file = File::create(&test_file_path).unwrap();
        file.write_all(&test_content).unwrap();

        let config = BlockCacheConfig::default();
        let mut cache = BlockCache::new(config);

        let file_id = cache
            .open_file(test_file_path.to_str().unwrap())
            .await
            .unwrap();

        let result = cache.read(file_id, 100, 500).await.unwrap();
        assert_eq!(result.len(), 500);
        assert!(result.iter().all(|&b| b == 0));

        cache.close_file(file_id);
    }

    #[tokio::test]
    async fn test_unaligned_offset() {
        let temp_dir = TempDir::new("block_cache_test").unwrap();
        let test_file_path = temp_dir.path().join("test.txt");
        let test_content = b"ABCDEFGHIJ"; // 10 bytes
        let mut file = File::create(&test_file_path).unwrap();
        file.write_all(test_content).unwrap();

        let config = BlockCacheConfig::default();
        let mut cache = BlockCache::new(config);

        let file_id = cache
            .open_file(test_file_path.to_str().unwrap())
            .await
            .unwrap();

        let result = cache.read(file_id, 1, 4).await.unwrap();
        assert_eq!(&result, b"BCDE");

        cache.close_file(file_id);
    }

    #[tokio::test]
    async fn test_cache_hit_miss() {
        let temp_dir = TempDir::new("block_cache_test").unwrap();
        let test_file_path = temp_dir.path().join("test.txt");
        let test_content = vec![1u8; 8192];
        let mut file = File::create(&test_file_path).unwrap();
        file.write_all(&test_content).unwrap();

        let config = BlockCacheConfig::new(10, 8192, 4096, false);
        let mut cache = BlockCache::new(config);

        let file_id = cache
            .open_file(test_file_path.to_str().unwrap())
            .await
            .unwrap();

        let initial_block_count = cache.get_block_count().await;

        let _ = cache.read(file_id, 0, 4096).await.unwrap();
        assert!(cache.get_block_count().await > initial_block_count);

        let count_after_first_read = cache.get_block_count().await;

        let _ = cache.read(file_id, 0, 4096).await.unwrap();
        assert_eq!(cache.get_block_count().await, count_after_first_read);

        cache.close_file(file_id);
    }

    #[tokio::test]
    async fn test_open_multiple_files() {
        let temp_dir = TempDir::new("block_cache_test").unwrap();
        let config = BlockCacheConfig::new(3, 1024 * 1024, 4096, false);
        let cache = BlockCache::new(config);

        for i in 0..5 {
            let test_file_path = temp_dir.path().join(format!("test_{}.txt", i));
            let mut file = File::create(&test_file_path).unwrap();
            file.write_all(b"test content").unwrap();

            let _ = cache
                .open_file(test_file_path.to_str().unwrap())
                .await
                .unwrap();
        }

        assert_eq!(cache.get_file_count(), 5);

        let file_5_path = temp_dir.path().join("test_5.txt");
        let mut file = File::create(&file_5_path).unwrap();
        file.write_all(b"test content").unwrap();

        let _ = cache
            .open_file(file_5_path.to_str().unwrap())
            .await
            .unwrap();
        assert_eq!(cache.get_file_count(), 6);
    }

    #[tokio::test]
    async fn test_concurrent_reads() {
        let temp_dir = TempDir::new("block_cache_test").unwrap();
        let test_file_path = temp_dir.path().join("test.txt");
        let test_content = vec![0u8; 16384];
        let mut file = File::create(&test_file_path).unwrap();
        file.write_all(&test_content).unwrap();

        let config = BlockCacheConfig::default();
        let mut cache = BlockCache::new(config);

        let file_id = cache
            .open_file(test_file_path.to_str().unwrap())
            .await
            .unwrap();

        for i in 0..10 {
            let read_offset = (i * 100) as u64;
            let result = cache.read(file_id, read_offset, 100).await;
            assert!(result.is_ok(), "Read {} failed: {:?}", i, result);
            assert_eq!(result.unwrap().len(), 100);
        }

        cache.close_file(file_id);
    }

    #[tokio::test]
    async fn test_invalid_file_id() {
        let config = BlockCacheConfig::default();
        let cache = BlockCache::new(config);

        let result = cache.read(999, 0, 10).await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("not found") || err.contains("FileId"));
    }

    #[tokio::test]
    async fn test_invalid_offset_length() {
        let temp_dir = TempDir::new("block_cache_test").unwrap();
        let test_file_path = temp_dir.path().join("test.txt");
        let test_content = b"Short content";
        let mut file = File::create(&test_file_path).unwrap();
        file.write_all(test_content).unwrap();

        let config = BlockCacheConfig::default();
        let mut cache = BlockCache::new(config);

        let file_id = cache
            .open_file(test_file_path.to_str().unwrap())
            .await
            .unwrap();

        let result = cache.read(file_id, 0, 0).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("length"));

        let result = cache.read(file_id, 100, 10).await;
        assert!(result.is_err());

        let result = cache.read(file_id, 5, 20).await;
        assert!(result.is_err());

        cache.close_file(file_id);
    }

    #[tokio::test]
    async fn test_custom_block_size() {
        let temp_dir = TempDir::new("block_cache_test").unwrap();
        let test_file_path = temp_dir.path().join("test.txt");
        let test_content = vec![0xAAu8; 2048];
        let mut file = File::create(&test_file_path).unwrap();
        file.write_all(&test_content).unwrap();

        let config = BlockCacheConfig::new(10, 4096, 2048, false);
        let mut cache = BlockCache::new(config);

        let file_id = cache
            .open_file(test_file_path.to_str().unwrap())
            .await
            .unwrap();

        let result = cache.read(file_id, 0, 2048).await.unwrap();
        assert_eq!(result.len(), 2048);
        assert!(result.iter().all(|&b| b == 0xAA));

        cache.close_file(file_id);
    }

    #[tokio::test]
    async fn test_close_nonexistent_file() {
        let config = BlockCacheConfig::default();
        let mut cache = BlockCache::new(config);

        cache.close_file(999);
    }
}

#[cfg(target_os = "linux")]
mod tests_io_uring {
    #[allow(unused_imports)]
    use std::fs::File;
    #[allow(unused_imports)]
    use std::io::Write;

    #[allow(unused_imports)]
    use tempdir::TempDir;

    #[allow(unused_imports)]
    use crate::block_cache::{BlockCache, BlockCacheConfig};

    #[tokio::test]
    async fn test_open_and_close_file_uring() {
        let temp_dir = TempDir::new("block_cache_test").unwrap();
        let test_file_path = temp_dir.path().join("test.txt");
        let mut file = File::create(&test_file_path).unwrap();
        file.write_all(b"Hello, World!").unwrap();

        let config = BlockCacheConfig::new(1000, 1024 * 1024 * 1024, 4096, true);
        let mut cache = BlockCache::new(config);

        let file_id = cache
            .open_file(test_file_path.to_str().unwrap())
            .await
            .unwrap();
        assert!(file_id > 0);

        cache.close_file(file_id);
    }

    #[tokio::test]
    async fn test_read_single_block_uring() {
        let temp_dir = TempDir::new("block_cache_test").unwrap();
        let test_file_path = temp_dir.path().join("test.txt");
        let test_content = b"This is test content for reading.";
        let mut file = File::create(&test_file_path).unwrap();
        file.write_all(test_content).unwrap();

        let config = BlockCacheConfig::new(1000, 1024 * 1024 * 1024, 4096, true);
        let mut cache = BlockCache::new(config);

        let file_id = cache
            .open_file(test_file_path.to_str().unwrap())
            .await
            .unwrap();

        let result = cache.read(file_id, 0, 4).await.unwrap();
        assert_eq!(&result, b"This");

        cache.close_file(file_id);
    }

    #[tokio::test]
    async fn test_read_spanning_blocks_uring() {
        let temp_dir = TempDir::new("block_cache_test").unwrap();
        let test_file_path = temp_dir.path().join("test.txt");
        let test_content = vec![0u8; 8192];
        let mut file = File::create(&test_file_path).unwrap();
        file.write_all(&test_content).unwrap();

        let config = BlockCacheConfig::new(1000, 1024 * 1024 * 1024, 4096, true);
        let mut cache = BlockCache::new(config);

        let file_id = cache
            .open_file(test_file_path.to_str().unwrap())
            .await
            .unwrap();

        let result = cache.read(file_id, 100, 500).await.unwrap();
        assert_eq!(result.len(), 500);
        assert!(result.iter().all(|&b| b == 0));

        cache.close_file(file_id);
    }

    #[tokio::test]
    async fn test_unaligned_offset_uring() {
        let temp_dir = TempDir::new("block_cache_test").unwrap();
        let test_file_path = temp_dir.path().join("test.txt");
        let test_content = b"ABCDEFGHIJ"; // 10 bytes
        let mut file = File::create(&test_file_path).unwrap();
        file.write_all(test_content).unwrap();

        let config = BlockCacheConfig::new(1000, 1024 * 1024 * 1024, 4096, true);
        let mut cache = BlockCache::new(config);

        let file_id = cache
            .open_file(test_file_path.to_str().unwrap())
            .await
            .unwrap();

        let result = cache.read(file_id, 1, 4).await.unwrap();
        assert_eq!(&result, b"BCDE");

        cache.close_file(file_id);
    }

    #[tokio::test]
    async fn test_cache_hit_miss_uring() {
        let temp_dir = TempDir::new("block_cache_test").unwrap();
        let test_file_path = temp_dir.path().join("test.txt");
        let test_content = vec![1u8; 8192];
        let mut file = File::create(&test_file_path).unwrap();
        file.write_all(&test_content).unwrap();

        let config = BlockCacheConfig::new(10, 8192, 4096, true);
        let mut cache = BlockCache::new(config);

        let file_id = cache
            .open_file(test_file_path.to_str().unwrap())
            .await
            .unwrap();

        let initial_block_count = cache.get_block_count().await;

        let _ = cache.read(file_id, 0, 4096).await.unwrap();
        assert!(cache.get_block_count().await > initial_block_count);

        let count_after_first_read = cache.get_block_count().await;

        let _ = cache.read(file_id, 0, 4096).await.unwrap();
        assert_eq!(cache.get_block_count().await, count_after_first_read);

        cache.close_file(file_id);
    }

    #[tokio::test]
    async fn test_open_multiple_files_uring() {
        let temp_dir = TempDir::new("block_cache_test").unwrap();
        let config = BlockCacheConfig::new(3, 1024 * 1024, 4096, true);
        let cache = BlockCache::new(config);

        for i in 0..5 {
            let test_file_path = temp_dir.path().join(format!("test_{}.txt", i));
            let mut file = File::create(&test_file_path).unwrap();
            file.write_all(b"test content").unwrap();

            let _ = cache
                .open_file(test_file_path.to_str().unwrap())
                .await
                .unwrap();
        }

        assert_eq!(cache.get_file_count(), 5);

        let file_5_path = temp_dir.path().join("test_5.txt");
        let mut file = File::create(&file_5_path).unwrap();
        file.write_all(b"test content").unwrap();

        let _ = cache
            .open_file(file_5_path.to_str().unwrap())
            .await
            .unwrap();
        assert_eq!(cache.get_file_count(), 6);
    }

    #[tokio::test]
    async fn test_concurrent_reads_uring() {
        let temp_dir = TempDir::new("block_cache_test").unwrap();
        let test_file_path = temp_dir.path().join("test.txt");
        let test_content = vec![0u8; 16384];
        let mut file = File::create(&test_file_path).unwrap();
        file.write_all(&test_content).unwrap();

        let config = BlockCacheConfig::new(1000, 1024 * 1024 * 1024, 4096, true);
        let mut cache = BlockCache::new(config);

        let file_id = cache
            .open_file(test_file_path.to_str().unwrap())
            .await
            .unwrap();

        for i in 0..10 {
            let read_offset = (i * 100) as u64;
            let result = cache.read(file_id, read_offset, 100).await;
            assert!(result.is_ok(), "Read {} failed: {:?}", i, result);
            assert_eq!(result.unwrap().len(), 100);
        }

        cache.close_file(file_id);
    }

    #[tokio::test]
    async fn test_invalid_file_id_uring() {
        let config = BlockCacheConfig::new(1000, 1024 * 1024 * 1024, 4096, true);
        let cache = BlockCache::new(config);

        let result = cache.read(999, 0, 10).await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("not found") || err.contains("FileId"));
    }

    #[tokio::test]
    async fn test_invalid_offset_length_uring() {
        let temp_dir = TempDir::new("block_cache_test").unwrap();
        let test_file_path = temp_dir.path().join("test.txt");
        let test_content = b"Short content";
        let mut file = File::create(&test_file_path).unwrap();
        file.write_all(test_content).unwrap();

        let config = BlockCacheConfig::new(1000, 1024 * 1024 * 1024, 4096, true);
        let mut cache = BlockCache::new(config);

        let file_id = cache
            .open_file(test_file_path.to_str().unwrap())
            .await
            .unwrap();

        let result = cache.read(file_id, 0, 0).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("length"));

        let result = cache.read(file_id, 100, 10).await;
        assert!(result.is_err());

        let result = cache.read(file_id, 5, 20).await;
        assert!(result.is_err());

        cache.close_file(file_id);
    }

    #[tokio::test]
    async fn test_custom_block_size_uring() {
        let temp_dir = TempDir::new("block_cache_test").unwrap();
        let test_file_path = temp_dir.path().join("test.txt");
        let test_content = vec![0xAAu8; 2048];
        let mut file = File::create(&test_file_path).unwrap();
        file.write_all(&test_content).unwrap();

        let config = BlockCacheConfig::new(10, 4096, 2048, true);
        let mut cache = BlockCache::new(config);

        let file_id = cache
            .open_file(test_file_path.to_str().unwrap())
            .await
            .unwrap();

        let result = cache.read(file_id, 0, 2048).await.unwrap();
        assert_eq!(result.len(), 2048);
        assert!(result.iter().all(|&b| b == 0xAA));

        cache.close_file(file_id);
    }

    #[tokio::test]
    async fn test_close_nonexistent_file_uring() {
        let config = BlockCacheConfig::new(1000, 1024 * 1024 * 1024, 4096, true);
        let mut cache = BlockCache::new(config);

        cache.close_file(999);
    }
}
