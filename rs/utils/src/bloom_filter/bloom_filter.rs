use std::collections::hash_map::DefaultHasher;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::{BufWriter, Write};

use anyhow::Result;
use bitvec::prelude::*;

use crate::io::wrap_write;

#[derive(Hash)]
pub struct HashKey {
    user_id: u128,
    doc_id: u128,
}

pub struct InMemoryBloomFilter<T: ?Sized> {
    bits: BitVec<u64>,
    num_hash_functions: usize,
    _marker: std::marker::PhantomData<T>,
}

impl<T: ?Sized + Hash> InMemoryBloomFilter<T> {
    pub fn new(expected_elements: usize, false_positive_prob: f64) -> Self {
        // Calculate optimal parameters
        let m = -((expected_elements as f64) * false_positive_prob.ln()) / (2.0_f64.ln().powi(2));
        let m = m.ceil() as usize;

        let k = (m as f64 / expected_elements as f64 * 2.0_f64.ln()).ceil() as usize;

        let mut bits = BitVec::with_capacity(m);
        // Ensure bits is filled with false initially
        bits.resize(m, false);

        Self {
            bits,
            num_hash_functions: k,
            _marker: std::marker::PhantomData,
        }
    }

    fn combine_hashes(&self, hash1: u64, hash2: u64, seed: u64) -> usize {
        ((hash1.wrapping_add(seed.wrapping_mul(hash2))) % self.bits.len() as u64) as usize
    }

    /// Double hashing for more random, uniform distribution of bits
    fn double_hash(&self, item: &T, seed: u64) -> usize {
        // Two different hash functions
        let mut hasher1 = DefaultHasher::new();
        let mut hasher2 = DefaultHasher::new();

        // Use different seeds for different hash functions
        hasher1.write_u64(seed);
        hasher2.write_u64(seed.wrapping_mul(0x9e3779b9)); // A common technique for seed mixing

        item.hash(&mut hasher1);
        item.hash(&mut hasher2);

        let hash1 = hasher1.finish();
        let hash2 = hasher2.finish();

        // Combine and map to bit vector size
        self.combine_hashes(hash1, hash2, seed)
    }

    pub fn insert(&mut self, item: &T) {
        for i in 0..self.num_hash_functions {
            let bit_index = self.double_hash(item, i as u64);
            self.bits.set(bit_index, true);
        }
    }

    pub fn may_contain(&self, item: &T) -> bool {
        (0..self.num_hash_functions).all(|i| self.bits[self.double_hash(item, i as u64)])
    }

    pub fn clear(&mut self) {
        self.bits.fill(false);
    }

    pub fn persist(&self, file_path: String) -> Result<usize> {
        let mut file = File::create(file_path)?;

        // Create buffered writer with mutable file reference
        let mut writer = BufWriter::new(&mut file);

        // Write header (m and k as u64)
        let mut total_bytes_written =
            wrap_write(&mut writer, &((self.bits.len() as u64).to_le_bytes()))?;
        total_bytes_written += wrap_write(
            &mut writer,
            &((self.num_hash_functions as u64).to_le_bytes()),
        )?;
        // Write bit vector
        let bits: &[u64] = self.bits.as_raw_slice();
        for &val in bits.iter() {
            total_bytes_written += wrap_write(&mut writer, &val.to_le_bytes())?;
        }

        writer.flush()?;

        Ok(total_bytes_written)
    }
}

#[cfg(test)]
mod tests {
    use std::io::{BufReader, Read};

    use tempdir::TempDir;

    use super::*;

    #[test]
    fn test_bloom_filter() {
        // Create a Bloom filter up to 4 billion documents and 1% false positive rate
        let mut filter = InMemoryBloomFilter::<HashKey>::new(4_000_000_000, 0.01);

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

        let mut str_filter = InMemoryBloomFilter::<str>::new(4_000_000_000, 0.01);
        str_filter.insert("hello");
        assert!(str_filter.may_contain("hello"));
        assert!(!str_filter.may_contain("world"));

        let mut u64_filter = InMemoryBloomFilter::<u64>::new(4_000_000_000, 0.01);
        u64_filter.insert(&42);
        assert!(u64_filter.may_contain(&42));
        assert!(!u64_filter.may_contain(&41));
    }

    #[test]
    fn test_combine_hashes_overflow() {
        let filter = InMemoryBloomFilter::<u64>::new(100, 0.01);

        // Test with values known to cause overflow
        let result = filter.combine_hashes(u64::MAX, u64::MAX, u64::MAX);

        // The test passes if this doesn't panic
        assert!(result < filter.bits.len());
    }

    #[test]
    fn test_bloom_filter_persist() {
        let temp_dir = TempDir::new("test_bloom_filter_persist")
            .expect("Failed to create temporary directory");
        let base_directory = temp_dir
            .path()
            .to_str()
            .expect("Failed to convert temporary directory path to string")
            .to_string();

        // Create a Bloom filter with expected elements and false positive probability
        let mut filter = InMemoryBloomFilter::<str>::new(100_000, 0.01);

        // Insert an element into the Bloom filter
        let test_item = "test_element";
        filter.insert(&test_item);

        // Persist the Bloom filter to disk
        let file_path = format!("{}/bloom-filter", base_directory);
        let total_bytes_written = filter
            .persist(file_path.clone())
            .expect("Failed to persist bloom filter to file");

        let file = File::open(&file_path).expect("Failed to open persisted file");
        let mut reader = BufReader::new(file);

        // Read header: number of bits and hash functions
        let mut buffer = [0u8; 8];
        assert!(reader.read_exact(&mut buffer).is_ok());
        let num_bits = u64::from_le_bytes(buffer) as usize;

        assert!(reader.read_exact(&mut buffer).is_ok());
        let num_hash_functions = u64::from_le_bytes(buffer) as usize;

        // Verify that the metadata matches
        assert_eq!(num_bits, filter.bits.len());
        assert_eq!(num_hash_functions, filter.num_hash_functions);
        let expected_num_u64_elements = (num_bits + 63) / 64;
        // Verify that the raw slice length matches the expected padded length
        assert_eq!(filter.bits.as_raw_slice().len(), expected_num_u64_elements,);

        // Read the raw bit vector data
        let mut raw_bits = vec![0u64; filter.bits.as_raw_slice().len()];
        for raw_bit in &mut raw_bits {
            let mut bit_buffer = [0u8; 8];
            assert!(reader.read_exact(&mut bit_buffer).is_ok());
            *raw_bit = u64::from_le_bytes(bit_buffer);
        }

        assert_eq!(&raw_bits[..], filter.bits.as_raw_slice());

        // Verify the total bytes written matches the expected size
        let expected_total_bytes =
            8 + 8 + (filter.bits.as_raw_slice().len() * std::mem::size_of::<u64>()) as usize;
        assert_eq!(total_bytes_written, expected_total_bytes);
    }
}
