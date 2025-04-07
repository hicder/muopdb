use std::hash::Hash;

use anyhow::Result;

use crate::bloom_filter::bloom_filter::InMemoryBloomFilter;

pub struct BloomFilterBuilder {
    base_directory: String,
    expected_elements: usize,
    false_positive_prob: f64,
}

impl BloomFilterBuilder {
    pub fn new(base_directory: String, expected_elements: usize, false_positive_prob: f64) -> Self {
        Self {
            base_directory,
            expected_elements,
            false_positive_prob,
        }
    }

    /// Builds and persists the Bloom filter to disk
    pub fn build_and_persist<T: ?Sized + Hash>(&self) -> Result<()> {
        let filter: InMemoryBloomFilter<T> =
            InMemoryBloomFilter::new(self.expected_elements, self.false_positive_prob);
        filter.persist(format!("{}/bloom-filter", self.base_directory))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::{BufReader, Read};

    use tempdir::TempDir;

    use super::*;

    #[test]
    fn test_bloom_filter_builder() {
        // Temporary file path for testing
        let temp_dir = TempDir::new("test_bloom_filter_builder")
            .expect("Failed to create temporary directory");
        let base_directory = temp_dir
            .path()
            .to_str()
            .expect("Failed to convert temporary directory path to string")
            .to_string();

        // Create a builder with expected elements and false positive probability
        let expected_elements = 100_000;
        let false_positive_prob = 0.01;
        let builder = BloomFilterBuilder::new(
            base_directory.clone(),
            expected_elements,
            false_positive_prob,
        );

        // Build and persist the Bloom filter
        assert!(builder.build_and_persist::<str>().is_ok());

        // Verify that the file was created
        let file_path = format!("{}/bloom-filter", base_directory);
        assert!(std::fs::metadata(&file_path).is_ok());

        // Open the file and verify its contents
        let file = File::open(&file_path).expect("Failed to open persisted file");
        let mut reader = BufReader::new(file);

        // Read header: number of bits and number of hash functions
        let mut buffer = [0u8; 8];
        assert!(reader.read_exact(&mut buffer).is_ok());
        let num_bits = u64::from_le_bytes(buffer) as usize;

        assert!(reader.read_exact(&mut buffer).is_ok());
        let num_hash_functions = u64::from_le_bytes(buffer) as usize;

        assert_eq!(num_bits, 958506);
        assert_eq!(num_hash_functions, 7);
    }
}
