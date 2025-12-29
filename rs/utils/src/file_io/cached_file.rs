use std::fs::Metadata;
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;

use crate::block_cache::cache::{BlockCache, FileId};
use crate::file_io::FileIO;

/// A [`FileIO`] implementation that uses a [`BlockCache`] for buffered reading.
///
/// This provides a transparent caching layer over any file, using the
/// global or private block cache for performance.
pub struct CachedFileIO {
    block_cache: Arc<BlockCache>,
    file_id: FileId,
}

impl CachedFileIO {
    /// Creates a new `CachedFileIO` instance by opening a file through the block cache.
    ///
    /// # Arguments
    /// * `block_cache` - The block cache to use for buffering
    /// * `path` - The system path to the file
    ///
    /// # Returns
    /// A `Result` containing the `CachedFileIO` instance.
    pub async fn new(block_cache: Arc<BlockCache>, path: &str) -> Result<Self> {
        let file_id = block_cache.open_file(path).await?;
        Ok(Self {
            block_cache,
            file_id,
        })
    }
}

#[async_trait]
impl FileIO for CachedFileIO {
    /// Reads a contiguous byte range using the block cache.
    async fn read(&self, offset: u64, length: u64) -> Result<Vec<u8>> {
        self.block_cache.read(self.file_id, offset, length).await
    }

    /// Retrieves file metadata from the underlying file.
    async fn metadata(&self) -> Result<Metadata> {
        self.block_cache.metadata(self.file_id).await
    }

    /// Returns the length of the file in bytes.
    async fn file_length(&self) -> Result<u64> {
        self.block_cache.file_length(self.file_id).await
    }
}

impl Drop for CachedFileIO {
    fn drop(&mut self) {
        self.block_cache.close_file(self.file_id);
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::Write;

    use tempdir::TempDir;

    use super::*;
    use crate::block_cache::cache::BlockCacheConfig;

    #[tokio::test]
    async fn test_cached_file_read_basic() -> Result<()> {
        let temp_dir = TempDir::new("cached_file_test")?;
        let file_path = temp_dir.path().join("test_cached.txt");
        let data = b"hello cached world";
        {
            let mut file = File::create(&file_path)?;
            file.write_all(data)?;
            file.flush()?;
        }

        let config = BlockCacheConfig::default();
        let block_cache = Arc::new(BlockCache::new(config));
        let cached_io = CachedFileIO::new(block_cache, file_path.to_str().unwrap()).await?;

        // Read "hello"
        let buf = cached_io.read(0, 5).await?;
        assert_eq!(buf, b"hello");

        // Read "world"
        let buf = cached_io.read(13, 5).await?;
        assert_eq!(buf, b"world");

        assert_eq!(cached_io.file_length().await?, data.len() as u64);

        Ok(())
    }

    #[tokio::test]
    async fn test_cached_file_caching_behavior() -> Result<()> {
        let temp_dir = TempDir::new("cached_file_test")?;
        let file_path = temp_dir.path().join("test_behavior.txt");
        let data = vec![0u8; 8192];
        {
            let mut file = File::create(&file_path)?;
            file.write_all(&data)?;
            file.flush()?;
        }

        let config = BlockCacheConfig::new(10, 1024 * 1024, 4096, false);
        let block_cache = Arc::new(BlockCache::new(config));
        let cached_io = CachedFileIO::new(block_cache.clone(), file_path.to_str().unwrap()).await?;

        let initial_blocks = block_cache.get_block_count().await;

        // First read should populate cache
        let _ = cached_io.read(0, 4096).await?;
        let blocks_after_read = block_cache.get_block_count().await;
        assert!(blocks_after_read > initial_blocks);

        // Second read should be from cache
        let _ = cached_io.read(0, 4096).await?;
        assert_eq!(block_cache.get_block_count().await, blocks_after_read);

        Ok(())
    }

    #[tokio::test]
    async fn test_cached_file_metadata() -> Result<()> {
        let temp_dir = TempDir::new("cached_file_test")?;
        let file_path = temp_dir.path().join("test_metadata.txt");
        let data = b"metadata test";
        {
            let mut file = File::create(&file_path)?;
            file.write_all(data)?;
            file.flush()?;
        }

        let config = BlockCacheConfig::default();
        let block_cache = Arc::new(BlockCache::new(config));
        let cached_io = CachedFileIO::new(block_cache, file_path.to_str().unwrap()).await?;

        assert_eq!(cached_io.file_length().await?, data.len() as u64);
        let metadata = cached_io.metadata().await?;
        assert_eq!(metadata.len(), data.len() as u64);

        Ok(())
    }
}
