use anyhow::{anyhow, Result};

use crate::bloom_filter::immutable_bloom_filter::ImmutableBloomFilter;
use crate::bloom_filter::HEADER_SIZE;

pub struct BloomFilterReader {
    base_directory: String,
}

impl BloomFilterReader {
    pub fn new(base_directory: String) -> Self {
        Self { base_directory }
    }

    pub async fn read_async(
        &self,
        env: std::sync::Arc<dyn crate::file_io::env::Env>,
    ) -> Result<ImmutableBloomFilter> {
        let path = format!("{}/bloom_filter", self.base_directory);
        let open_result = env.open(&path).await?;
        let file_io = open_result.file_io;

        let file_len = file_io.file_length().await?;
        if file_len < HEADER_SIZE as u64 {
            return Err(anyhow!("File too small to be a valid Bloom filter"));
        }

        let metadata = file_io.read(0, HEADER_SIZE as u64).await?;
        let num_hash_functions = u64::from_le_bytes(metadata[8..16].try_into()?) as usize;
        let num_blocks = u64::from_le_bytes(metadata[16..HEADER_SIZE].try_into()?) as usize;

        Ok(ImmutableBloomFilter::new(
            file_io,
            HEADER_SIZE,
            num_blocks,
            num_hash_functions,
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use tempdir::TempDir;

    use super::*;
    use crate::bloom_filter::blocked_bloom_filter::BlockedBloomFilter;
    use crate::bloom_filter::writer::BloomFilterWriter;
    use crate::bloom_filter::BloomFilter;
    use crate::file_io::env::{DefaultEnv, EnvConfig};

    #[tokio::test]
    async fn test_bloom_filter_reader() {
        let temp_dir =
            TempDir::new("test_bloom_filter_reader").expect("Failed to create temporary directory");
        let base_directory = temp_dir
            .path()
            .to_str()
            .expect("Failed to convert temporary directory path to string")
            .to_string();
        let writer = BloomFilterWriter::new(base_directory.clone());

        let mut bloom_filter = BlockedBloomFilter::new(1000, 0.01);
        bloom_filter.insert("test_key");
        bloom_filter.insert("another_key");

        assert!(writer.write(&bloom_filter).is_ok());

        let reader = BloomFilterReader::new(base_directory.clone());
        let env_config = EnvConfig::default();
        let env = Arc::new(DefaultEnv::new(env_config));

        let mmap_filter = reader
            .read_async(env)
            .await
            .expect("Failed to read immutable bloom filter");

        assert!(mmap_filter.may_contain(&"test_key").await.unwrap());
        assert!(mmap_filter.may_contain(&"another_key").await.unwrap());
        assert!(!mmap_filter.may_contain(&"missing_key").await.unwrap());
    }
}
