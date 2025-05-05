use anyhow::Result;
use memmap2::Mmap;

use crate::bloom_filter::{BloomFilter, BLOCK_SIZE_IN_BITS};

pub struct ImmutableBloomFilter {
    mmap: Mmap,
    data_offset: usize,
    num_blocks: usize,
    num_hash_functions: usize,
}

impl ImmutableBloomFilter {
    pub fn new(
        file_path: String,
        data_offset: usize,
        num_blocks: usize,
        num_hash_functions: usize,
    ) -> Result<Self> {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .open(file_path.clone())?;
        let mmap = unsafe { Mmap::map(&file) }?;

        Ok(Self {
            mmap,
            data_offset,
            num_blocks,
            num_hash_functions,
        })
    }
}

impl BloomFilter for ImmutableBloomFilter {
    fn num_hash_functions(&self) -> usize {
        self.num_hash_functions
    }

    fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    fn is_bit_set(&self, block_idx: usize, bit_pos_in_block: usize) -> bool {
        let block_start = self.data_offset + (block_idx * BLOCK_SIZE_IN_BITS / 8);
        let byte = self.mmap[block_start + bit_pos_in_block / 8];
        (byte & (1 << (bit_pos_in_block % 8))) != 0
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
    fn test_immutable_bloom_filter() {
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

        let mmap_filter = ImmutableBloomFilter::new(
            format!("{}/bloom_filter", base_directory),
            /* data_offset */ 24,
            bloom_filter.num_blocks(),
            bloom_filter.num_hash_functions(),
        )
        .expect("Failed to create immutable bloom filter");

        assert!(mmap_filter.may_contain(&"test_key"));
        assert!(mmap_filter.may_contain(&"another_key"));
        assert!(!mmap_filter.may_contain(&"missing_key"));
    }
}
