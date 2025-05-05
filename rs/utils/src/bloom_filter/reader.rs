use std::fs::File;

use anyhow::{anyhow, Result};
use memmap2::Mmap;

use crate::bloom_filter::immutable_bloom_filter::ImmutableBloomFilter;
use crate::bloom_filter::HEADER_SIZE;

pub struct BloomFilterReader {
    base_directory: String,
}

impl BloomFilterReader {
    pub fn new(base_directory: String) -> Self {
        Self { base_directory }
    }

    pub fn read(&self) -> Result<ImmutableBloomFilter> {
        let path = format!("{}/bloom_filter", self.base_directory);
        let file = File::open(&path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        if mmap.len() < HEADER_SIZE {
            return Err(anyhow!("File too small to be a valid Bloom filter"));
        }

        let metadata = &mmap[..HEADER_SIZE];
        let num_hash_functions = u64::from_le_bytes(metadata[8..16].try_into()?) as usize;
        let num_blocks = u64::from_le_bytes(metadata[16..HEADER_SIZE].try_into()?) as usize;

        Ok(ImmutableBloomFilter::new(
            path,
            HEADER_SIZE,
            num_blocks,
            num_hash_functions,
        )?)
    }
}

#[cfg(test)]
mod tests {
    use tempdir::TempDir;

    use super::*;
    use crate::bloom_filter::blocked_bloom_filter::BlockedBloomFilter;
    use crate::bloom_filter::writer::BloomFilterWriter;
    use crate::bloom_filter::BloomFilter;

    #[test]
    fn test_bloom_filter_reader() {
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
        let mmap_filter = reader
            .read()
            .expect("Failed to read immutable bloom filter");

        assert!(mmap_filter.may_contain(&"test_key"));
        assert!(mmap_filter.may_contain(&"another_key"));
        assert!(!mmap_filter.may_contain(&"missing_key"));
    }
}
