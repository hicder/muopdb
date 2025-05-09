#[allow(clippy::module_inception)]
pub mod blocked_bloom_filter;
pub mod immutable_bloom_filter;
pub mod reader;
pub mod writer;

use std::hash::{Hash, Hasher};

use xxhash_rust::xxh3::Xxh3;

/// Optimized block size: modern processors typically use 64-byte or 128-byte cache lines,
/// so having a block size of 512 bits (64 bytes) would minimize cache misses during lookups.
pub const BLOCK_SIZE_IN_BITS: usize = 512;

/// 3 u64 values:
/// - block_size_in_bits
/// - num_hash_functions
/// - num_blocks
pub const HEADER_SIZE: usize = 24;

#[derive(Copy, Clone)]
pub struct HashIdx {
    // h1 (lower 32 bits of XXH3) selects the block
    // h2 (upper 32 bits of XXH3) sets bits within the block
    //
    // Rationale:
    // - lower bits are more chaotic, reducing collisions between
    // keys mapping to the same block.
    // - upper bits are more stable across hash computations,
    // ensuring uniform bit distribution within a block.
    h1: u32,
    h2: u32,
}

pub trait BloomFilter {
    // Required methods (implementors must provide these)
    fn num_blocks(&self) -> usize;
    fn num_hash_functions(&self) -> usize;

    /// Implementors must define how to check if a bit is set at `bit_pos_in_block`.
    fn is_bit_set(&self, block_idx: usize, bit_pos_in_block: usize) -> bool;

    // Provided helper methods (shared logic)
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
        (h1 as usize).wrapping_mul(self.num_blocks()) >> 32
    }

    /// Fully shared `check_bits` implementation.
    fn check_bits(&self, mut h: u32, block_idx: usize) -> bool {
        for _ in 0..self.num_hash_functions() {
            let bit_pos_in_block = (h >> (32 - BLOCK_SIZE_IN_BITS.trailing_zeros())) as usize;
            if !self.is_bit_set(block_idx, bit_pos_in_block) {
                return false;
            }
            h = h.wrapping_mul(0x9e3779b9);
        }
        true
    }

    /// Fully shared `may_contain` implementation.
    fn may_contain<T: Hash + ?Sized>(&self, key: &T) -> bool {
        let hash_idx = self.hash_key(key);
        let block_idx = self.get_block_idx(hash_idx.h1);
        self.check_bits(hash_idx.h2, block_idx)
    }
}
