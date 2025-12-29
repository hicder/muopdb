use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use dashmap::DashMap;

use crate::block_cache::cache::{BlockCache, BlockCacheConfig, FileId};
use crate::file_io::cached_file::CachedFileIO;
use crate::file_io::mmap_file::MMapFileIO;
use crate::file_io::FileIO;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileType {
    MMap,
    CachedStandard,
    #[cfg(target_os = "linux")]
    CachedIoUring,
}

#[derive(Debug, Clone)]
pub struct EnvConfig {
    pub file_type: FileType,
    pub block_size: usize,
    pub block_cache_capacity_bytes: u64,
    pub max_open_files: u64,
}

impl Default for EnvConfig {
    fn default() -> Self {
        Self {
            file_type: FileType::CachedStandard,
            block_size: 4096,
            block_cache_capacity_bytes: 1024 * 1024 * 1024, // 1GB
            max_open_files: 1024,
        }
    }
}

pub struct OpenResult {
    pub file_id: FileId,
    pub file_io: Arc<dyn FileIO + Send + Sync>,
}

#[async_trait]
pub trait Env: Send + Sync {
    async fn open(&self, path: &str) -> Result<OpenResult>;
    async fn close(&self, file_id: FileId) -> Result<()>;
}

pub struct DefaultEnv {
    config: EnvConfig,
    block_cache: Option<Arc<BlockCache>>,
    file_id_generator: AtomicU64,
    open_files: DashMap<FileId, Arc<dyn FileIO + Send + Sync>>,
}

impl DefaultEnv {
    pub fn new(config: EnvConfig) -> Self {
        let block_cache = match config.file_type {
            FileType::MMap => None,
            FileType::CachedStandard => {
                let cache_config = BlockCacheConfig::new(
                    config.max_open_files,
                    config.block_cache_capacity_bytes,
                    config.block_size,
                    false,
                );
                Some(Arc::new(BlockCache::new(cache_config)))
            }
            #[cfg(target_os = "linux")]
            FileType::CachedIoUring => {
                let cache_config = BlockCacheConfig::new(
                    config.max_open_files,
                    config.block_cache_capacity_bytes,
                    config.block_size,
                    true,
                );
                Some(Arc::new(BlockCache::new(cache_config)))
            }
        };

        Self {
            config,
            block_cache,
            file_id_generator: AtomicU64::new(1),
            open_files: DashMap::new(),
        }
    }
}

#[async_trait]
impl Env for DefaultEnv {
    async fn open(&self, path: &str) -> Result<OpenResult> {
        let file_io: Arc<dyn FileIO + Send + Sync> = match self.config.file_type {
            FileType::MMap => Arc::new(MMapFileIO::new(path)?),
            FileType::CachedStandard => {
                let cache = self
                    .block_cache
                    .as_ref()
                    .ok_or_else(|| anyhow!("Block cache not initialized"))?;
                Arc::new(CachedFileIO::new(cache.clone(), path).await?)
            }
            #[cfg(target_os = "linux")]
            FileType::CachedIoUring => {
                let cache = self
                    .block_cache
                    .as_ref()
                    .ok_or_else(|| anyhow!("Block cache not initialized"))?;
                Arc::new(CachedFileIO::new(cache.clone(), path).await?)
            }
        };

        let file_id = self.file_id_generator.fetch_add(1, Ordering::SeqCst);
        self.open_files.insert(file_id, file_io.clone());

        Ok(OpenResult { file_id, file_io })
    }

    async fn close(&self, file_id: FileId) -> Result<()> {
        if self.open_files.remove(&file_id).is_none() {
            return Err(anyhow!("FileId {} not found", file_id));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::Write;

    use tempdir::TempDir;

    use super::*;

    async fn create_test_file(path: &std::path::Path, data: &[u8]) -> Result<()> {
        let mut file = File::create(path)?;
        file.write_all(data)?;
        file.flush()?;
        Ok(())
    }

    #[tokio::test]
    async fn test_default_env_mmap() -> Result<()> {
        let temp_dir = TempDir::new("env_test_mmap")?;
        let file_path = temp_dir.path().join("test.bin");
        let data = b"hello mmap env";
        create_test_file(&file_path, data).await?;

        let config = EnvConfig {
            file_type: FileType::MMap,
            ..EnvConfig::default()
        };
        let env = DefaultEnv::new(config);

        let res = env.open(file_path.to_str().unwrap()).await?;
        assert_eq!(res.file_io.read(0, 5).await?, b"hello");

        env.close(res.file_id).await?;
        assert!(env.close(res.file_id).await.is_err());

        Ok(())
    }

    #[tokio::test]
    async fn test_default_env_cached_standard() -> Result<()> {
        let temp_dir = TempDir::new("env_test_cached")?;
        let file_path = temp_dir.path().join("test_cached.bin");
        let data = b"hello cached env";
        create_test_file(&file_path, data).await?;

        let config = EnvConfig {
            file_type: FileType::CachedStandard,
            block_size: 1024,
            ..EnvConfig::default()
        };
        let env = DefaultEnv::new(config);

        let res = env.open(file_path.to_str().unwrap()).await?;
        assert_eq!(res.file_io.read(6, 6).await?, b"cached");

        env.close(res.file_id).await?;
        Ok(())
    }

    #[tokio::test]
    #[cfg(target_os = "linux")]
    async fn test_default_env_cached_uring() -> Result<()> {
        let temp_dir = TempDir::new("env_test_uring")?;
        let file_path = temp_dir.path().join("test_uring.bin");
        let data = b"hello uring env";
        create_test_file(&file_path, data).await?;

        let config = EnvConfig {
            file_type: FileType::CachedIoUring,
            ..EnvConfig::default()
        };
        let env = DefaultEnv::new(config);

        let res = env.open(file_path.to_str().unwrap()).await?;
        assert_eq!(res.file_io.read(12, 3).await?, b"env");

        env.close(res.file_id).await?;
        Ok(())
    }
}
