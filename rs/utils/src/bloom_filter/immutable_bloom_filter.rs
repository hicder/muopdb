use std::hash::{Hash, Hasher};

use anyhow::Result;
use memmap2::Mmap;
use xxhash_rust::xxh3::Xxh3;

use crate::bloom_filter::HashIdx;

pub struct ImmutableBloomFilter {
    mmap: Mmap,
    data_offset: usize,
    num_blocks: usize,
    num_hash_functions: usize,
    block_size_in_bits: usize,
}

impl ImmutableBloomFilter {
    pub fn new(
        file_path: String,
        data_offset: usize,
        num_blocks: usize,
        num_hash_functions: usize,
        block_size_in_bits: usize,
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
            block_size_in_bits,
        })
    }

    pub fn may_contain<T: Hash + ?Sized>(&self, key: &T) -> bool {
        let hash_idx = self.hash_key(key);
        let block_idx = self.get_block_idx(hash_idx.h1);
        self.check_bits(hash_idx.h2, block_idx)
    }

    fn hash_key<T: Hash + ?Sized>(&self, key: &T) -> HashIdx {
        let mut hasher = Xxh3::default();
        key.hash(&mut hasher);
        let hash = hasher.finish();
        HashIdx {
            h1: hash as u32,
            h2: (hash >> 32) as u32,
        }
    }

    fn get_block_idx(&self, h1: u32) -> usize {
        // This is just FastRange32
        (h1 as usize).wrapping_mul(self.num_blocks) >> 32
    }

    fn check_bits(&self, mut h: u32, block_idx: usize) -> bool {
        let block_start = self.data_offset + (block_idx * self.block_size_in_bits / 8);
        let block_end = block_start + self.block_size_in_bits / 8;
        let block = &self.mmap[block_start..block_end];

        for _ in 0..self.num_hash_functions {
            let bit_pos_in_block = (h >> (32 - self.block_size_in_bits.trailing_zeros())) as usize;
            let byte = block[bit_pos_in_block / 8];
            if (byte & (1 << (bit_pos_in_block % 8))) == 0 {
                return false;
            }
            h = h.wrapping_mul(0x9e3779b9);
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use tempdir::TempDir;

    use super::*;
    use crate::bloom_filter::blocked_bloom_filter::BlockedBloomFilter;
    use crate::bloom_filter::writer::BloomFilterWriter;

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
            BlockedBloomFilter::BLOCK_SIZE_IN_BITS,
        )
        .expect("Failed to create immutable bloom filter");

        assert!(mmap_filter.may_contain(&"test_key"));
        assert!(mmap_filter.may_contain(&"another_key"));
        assert!(!mmap_filter.may_contain(&"missing_key"));
    }
}
