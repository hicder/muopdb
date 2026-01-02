use std::sync::Arc;

use anyhow::Result;

use crate::bloom_filter::{BloomFilter, BLOCK_SIZE_IN_BITS};
use crate::file_io::FileIO;

pub struct ImmutableBloomFilter {
    file_io: Arc<dyn FileIO + Send + Sync>,
    data_offset: usize,
    num_blocks: usize,
    num_hash_functions: usize,
}

impl ImmutableBloomFilter {
    pub fn new(
        file_io: Arc<dyn FileIO + Send + Sync>,
        data_offset: usize,
        num_blocks: usize,
        num_hash_functions: usize,
    ) -> Self {
        Self {
            file_io,
            data_offset,
            num_blocks,
            num_hash_functions,
        }
    }
}

#[async_trait::async_trait]
impl BloomFilter for ImmutableBloomFilter {
    fn num_hash_functions(&self) -> usize {
        self.num_hash_functions
    }

    fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    async fn is_bit_set(&self, block_idx: usize, bit_pos_in_block: usize) -> Result<bool> {
        let block_start = self.data_offset + (block_idx * BLOCK_SIZE_IN_BITS / 8);
        let byte_pos = block_start + bit_pos_in_block / 8;
        let buf = self.file_io.read(byte_pos as u64, 1).await?;
        let byte = buf[0];
        Ok((byte & (1 << (bit_pos_in_block % 8))) != 0)
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
    use crate::file_io::standard_file::StandardFile;

    #[tokio::test]
    async fn test_immutable_bloom_filter() {
        let temp_dir = TempDir::new("test_immutable_bloom_filter")
            .expect("Failed to create temporary directory");
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

        let file_path = format!("{}/bloom_filter", base_directory);
        let file = tokio::fs::File::open(&file_path).await.unwrap();
        let standard_file = StandardFile::new(file).await;
        let file_io = Arc::new(standard_file);

        let mmap_filter = ImmutableBloomFilter::new(
            file_io,
            /* data_offset */ 24,
            bloom_filter.num_blocks(),
            bloom_filter.num_hash_functions(),
        );

        assert!(mmap_filter.may_contain(&"test_key").await.unwrap());
        assert!(mmap_filter.may_contain(&"another_key").await.unwrap());
        assert!(!mmap_filter.may_contain(&"missing_key").await.unwrap());
    }
}
