use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use moka::future::Cache;
use tokio::fs::{self, File};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tracing::{debug, warn};

/// Configuration for the disk cache.
#[derive(Debug, Clone)]
pub struct DiskCacheConfig {
    pub cache_dir: PathBuf,
    pub capacity_bytes: u64,
}

impl DiskCacheConfig {
    pub fn new(cache_dir: PathBuf, capacity_bytes: u64) -> Self {
        Self {
            cache_dir,
            capacity_bytes,
        }
    }
}

/// A block key for the disk cache, using the original path and block number.
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
struct DiskBlockKey {
    path: String,
    block_number: u64,
}

/// A local disk cache for blocks.
pub struct DiskCache {
    config: DiskCacheConfig,
    // Metadata cache: maps block key to its size.
    // We use moka to handle LRU and eviction of the files.
    metadata: Cache<DiskBlockKey, u64>,
}

impl DiskCache {
    pub fn new(config: DiskCacheConfig) -> Result<Self> {
        if !config.cache_dir.exists() {
            std::fs::create_dir_all(&config.cache_dir)?;
        }

        let cache_dir_clone = config.cache_dir.clone();

        let metadata = Cache::builder()
            .max_capacity(config.capacity_bytes)
            .weigher(|_key, &size| size as u32)
            .eviction_listener(move |key: Arc<DiskBlockKey>, size: u64, cause| {
                let cache_dir = cache_dir_clone.clone();
                tokio::spawn(async move {
                    debug!(
                        "[DISK_CACHE] Evicting: {} block {} (size: {}, cause: {:?})",
                        key.path, key.block_number, size, cause
                    );
                    let block_path = Self::compute_block_path(&cache_dir, &key);
                    if let Err(e) = fs::remove_file(&block_path).await {
                        warn!(
                            "[DISK_CACHE] Failed to remove evicted file {:?}: {}",
                            block_path, e
                        );
                    }
                });
            })
            .build();

        Ok(Self { config, metadata })
    }

    pub async fn get(&self, path: &str, block_number: u64) -> Result<Option<Vec<u8>>> {
        let key = DiskBlockKey {
            path: path.to_string(),
            block_number,
        };

        if !self.metadata.contains_key(&key) {
            // Check if it exists on disk but not in metadata (lazy load after restart)
            let block_path = Self::compute_block_path(&self.config.cache_dir, &key);
            if !block_path.exists() {
                return Ok(None);
            }
            // Populate metadata
            let attr = fs::metadata(&block_path).await?;
            self.metadata.insert(key.clone(), attr.len()).await;
        } else {
            // Touch it in moka to update LRU
            let _ = self.metadata.get(&key).await;
        }

        let block_path = Self::compute_block_path(&self.config.cache_dir, &key);
        let mut file = File::open(&block_path).await?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).await?;

        debug!("[DISK_CACHE] Hit: {} block {}", path, block_number);
        Ok(Some(buffer))
    }

    pub async fn put(&self, path: &str, block_number: u64, data: &[u8]) -> Result<()> {
        let key = DiskBlockKey {
            path: path.to_string(),
            block_number,
        };

        let block_path = Self::compute_block_path(&self.config.cache_dir, &key);
        if let Some(parent) = block_path.parent() {
            fs::create_dir_all(parent).await?;
        }

        // Write to a temp file first and rename to ensure atomicity?
        // For now, direct write is simpler.
        let mut file = File::create(&block_path).await?;
        file.write_all(data).await?;
        file.sync_all().await?;

        self.metadata.insert(key, data.len() as u64).await;

        // Wait for any pending evictions to be processed to keep disk in sync with metadata
        self.metadata.run_pending_tasks().await;

        Ok(())
    }

    fn compute_block_path(cache_dir: &Path, key: &DiskBlockKey) -> PathBuf {
        // s3://a/b/c.sst -> a/b/c.sst
        let sanitized = key.path.replace("s3://", "").replace("://", "_"); // fallback for other protocols

        cache_dir
            .join(sanitized)
            .join(format!("block_{:010}.bin", key.block_number))
    }
}
