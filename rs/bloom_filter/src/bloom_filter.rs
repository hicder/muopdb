use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use bitvec::prelude::*;

pub struct HashKey {
    user_id: u128,
    doc_id: u128,
}

pub struct BloomFilter {
    bits: BitVec,
    num_hash_functions: usize,
}

impl BloomFilter {
    pub fn new(expected_elements: usize, false_positive_prob: f64) -> Self {
        // Calculate optimal parameters
        let m = -((expected_elements as f64) * false_positive_prob.ln()) / (2.0_f64.ln().powi(2));
        let m = m.ceil() as usize;

        let k = (m as f64 / expected_elements as f64 * 2.0_f64.ln()).ceil() as usize;

        Self {
            bits: bitvec![0; m],
            num_hash_functions: k,
        }
    }

    /// Double hashing for more random, uniform distribution of bits
    fn double_hash(&self, item: &HashKey, seed: u64) -> usize {
        // Two different hash functions
        let mut hasher1 = DefaultHasher::new();
        let mut hasher2 = DefaultHasher::new();

        // Use different seeds for different hash functions
        hasher1.write_u64(seed);
        hasher2.write_u64(seed.wrapping_mul(0x9e3779b9)); // A common technique for seed mixing

        item.user_id.hash(&mut hasher1);
        item.doc_id.hash(&mut hasher1);
        item.user_id.hash(&mut hasher2);
        item.doc_id.hash(&mut hasher2);

        let hash1 = hasher1.finish();
        let hash2 = hasher2.finish();

        // Combine and map to bit vector size
        ((hash1 + seed * hash2) % self.bits.len() as u64) as usize
    }

    pub fn insert(&mut self, item: &HashKey) {
        for i in 0..self.num_hash_functions {
            let bit_index = self.double_hash(item, i as u64);
            self.bits.set(bit_index, true);
        }
    }

    pub fn may_contain(&self, item: &HashKey) -> bool {
        (0..self.num_hash_functions).all(|i| self.bits[self.double_hash(item, i as u64)])
    }

    pub fn clear(&mut self) {
        self.bits.fill(false);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bloom_filter() {
        // Create a Bloom filter up to 4 billion documents and 1% false positive rate
        let mut filter = BloomFilter::new(4_000_000_000, 0.01);

        let item1 = HashKey {
            user_id: 1u128,
            doc_id: 2u128,
        };
        let item2 = HashKey {
            user_id: 3u128,
            doc_id: 4u128,
        };

        filter.insert(&item1);
        assert!(filter.may_contain(&item1));
        assert!(!filter.may_contain(&item2));
    }
}
