use std::hash::{Hash, Hasher};

use bitvec::prelude::*;
use xxhash_rust::xxh3::Xxh3;

use crate::bloom_filter::HashIdx;

/// A blocked bloom filter improves upon the classing bloom filter by
/// 1. Dividing the bit vector into fixed-size blocks
/// 2. Mapping each key to exactly one block
/// 3. Performing all probes within that single block
///
/// Point 2 + 3 are key: each query only needs to read a single block.
pub struct BlockedBloomFilter {
    bits: BitVec<u8>,
    num_blocks: usize,
    num_hash_functions: usize,
}

impl BlockedBloomFilter {
    /// Another optimization: modern processors typically use 64-byte or 128-byte cache lines,
    /// so having a block size of 512 bits (64 bytes) would minimize cache misses during lookups.
    pub const BLOCK_SIZE_IN_BITS: usize = 512;

    pub fn new(expected_elements: usize, false_positive_prob: f64) -> Self {
        // Calculate optimal parameters
        let m = -((expected_elements as f64) * false_positive_prob.ln()) / (2.0_f64.ln().powi(2));
        let m = m.ceil() as usize;

        let k = (m as f64 / expected_elements as f64 * 2.0_f64.ln()).ceil() as usize;

        // Round up to BLOCK_SIZE_IN_BITS blocks
        let num_blocks = m.div_ceil(Self::BLOCK_SIZE_IN_BITS);

        // Ensure bits is filled with false initially
        let bits = bitvec![u8, Lsb0; 0; num_blocks * Self::BLOCK_SIZE_IN_BITS];

        Self {
            bits,
            num_blocks,
            num_hash_functions: k,
        }
    }

    pub fn insert<T: Hash + ?Sized>(&mut self, key: &T) {
        let hash_idx = self.hash_key(key);
        let block_idx = self.get_block_idx(hash_idx.h1);
        self.set_bits(hash_idx.h2, block_idx);
    }

    pub fn may_contain<T: Hash + ?Sized>(&self, key: &T) -> bool {
        let hash_idx = self.hash_key(key);
        let block_idx = self.get_block_idx(hash_idx.h1);
        self.check_bits(hash_idx.h2, block_idx)
    }

    pub fn bits(&self) -> &BitVec<u8> {
        &self.bits
    }

    pub fn num_hash_functions(&self) -> usize {
        self.num_hash_functions
    }

    pub fn num_blocks(&self) -> usize {
        self.num_blocks
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

    fn set_bits(&mut self, mut h: u32, block_idx: usize) {
        let block_offset = block_idx * Self::BLOCK_SIZE_IN_BITS;
        for _ in 0..self.num_hash_functions {
            let bit_pos_in_block = (h >> (32 - Self::BLOCK_SIZE_IN_BITS.trailing_zeros())) as usize;
            self.bits.set(block_offset + bit_pos_in_block, true);
            h = h.wrapping_mul(0x9e3779b9);
        }
    }

    fn check_bits(&self, mut h: u32, block_idx: usize) -> bool {
        let block_offset = block_idx * Self::BLOCK_SIZE_IN_BITS;
        for _ in 0..self.num_hash_functions {
            let bit_pos_in_block = (h >> (32 - Self::BLOCK_SIZE_IN_BITS.trailing_zeros())) as usize;
            if !self.bits[block_offset + bit_pos_in_block] {
                return false;
            }
            h = h.wrapping_mul(0x9e3779b9);
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blocked_bloom_filter() {
        // Create a Bloom filter up to 400 documents and 1% false positive rate
        let mut str_filter = BlockedBloomFilter::new(400, 0.01);
        str_filter.insert::<str>("hello");
        assert!(str_filter.may_contain::<str>("hello"));
        assert!(!str_filter.may_contain::<str>("world"));

        let mut u64_filter = BlockedBloomFilter::new(400, 0.01);
        u64_filter.insert::<u64>(&42);
        assert!(u64_filter.may_contain::<u64>(&42));
        assert!(!u64_filter.may_contain::<u64>(&41));
    }
}
