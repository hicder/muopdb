use std::hash::Hash;

use bitvec::prelude::*;

use crate::bloom_filter::{BloomFilter, BLOCK_SIZE_IN_BITS};

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
    pub fn new(expected_elements: usize, false_positive_prob: f64) -> Self {
        // Calculate optimal parameters
        let m = -((expected_elements as f64) * false_positive_prob.ln()) / (2.0_f64.ln().powi(2));
        let m = m.ceil() as usize;

        let k = (m as f64 / expected_elements as f64 * 2.0_f64.ln()).ceil() as usize;

        // Round up to BLOCK_SIZE_IN_BITS blocks
        let num_blocks = m.div_ceil(BLOCK_SIZE_IN_BITS);

        // Ensure bits is filled with false initially
        let bits = bitvec![u8, Lsb0; 0; num_blocks * BLOCK_SIZE_IN_BITS];

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

    pub fn bits(&self) -> &BitVec<u8> {
        &self.bits
    }

    fn set_bits(&mut self, mut h: u32, block_idx: usize) {
        let block_offset = block_idx * BLOCK_SIZE_IN_BITS;
        for _ in 0..self.num_hash_functions {
            let bit_pos_in_block = (h >> (32 - BLOCK_SIZE_IN_BITS.trailing_zeros())) as usize;
            self.bits.set(block_offset + bit_pos_in_block, true);
            h = h.wrapping_mul(0x9e3779b9);
        }
    }
}

impl BloomFilter for BlockedBloomFilter {
    fn num_hash_functions(&self) -> usize {
        self.num_hash_functions
    }

    fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    fn is_bit_set(&self, block_idx: usize, bit_pos_in_block: usize) -> bool {
        let block_offset = block_idx * BLOCK_SIZE_IN_BITS;
        self.bits[block_offset + bit_pos_in_block]
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
