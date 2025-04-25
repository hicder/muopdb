use std::fs::File;
use std::io::{BufWriter, Write};

use anyhow::Result;

use crate::bloom_filter::blocked_bloom_filter::BlockedBloomFilter;
use crate::io::wrap_write;

pub struct BloomFilterWriter {
    base_directory: String,
}

impl BloomFilterWriter {
    pub fn new(base_directory: String) -> Self {
        Self { base_directory }
    }

    pub fn write(&self, bloom_filter: &BlockedBloomFilter) -> Result<usize> {
        // Create file
        let path = format!("{}/bloom_filter", self.base_directory);
        let mut file = File::create(path)?;
        let mut writer = BufWriter::new(&mut file);

        // Write header (metadata)
        let mut bytes_written = wrap_write(
            &mut writer,
            &(BlockedBloomFilter::BLOCK_SIZE_IN_BITS as u64).to_le_bytes(),
        )?;
        bytes_written += wrap_write(
            &mut writer,
            &(bloom_filter.num_hash_functions() as u64).to_le_bytes(),
        )?;
        bytes_written += wrap_write(
            &mut writer,
            &(bloom_filter.bits().len() as u64).to_le_bytes(),
        )?;

        // Write bit vector
        let bits: &[u8] = bloom_filter.bits().as_raw_slice();
        for &val in bits.iter() {
            bytes_written += wrap_write(&mut writer, &val.to_le_bytes())?;
        }

        writer.flush()?;
        Ok(bytes_written)
    }
}

// Test
#[cfg(test)]
mod tests {
    use std::fs;
    use std::io::{BufReader, Read};

    use tempdir::TempDir;

    use super::*;

    #[test]
    fn test_bloom_filter_writer() {
        let temp_dir =
            TempDir::new("test_bloom_filter_writer").expect("Failed to create temporary directory");
        let base_directory = temp_dir
            .path()
            .to_str()
            .expect("Failed to convert temporary directory path to string")
            .to_string();
        let writer = BloomFilterWriter::new(base_directory.clone());

        // Create a test Bloom filter
        let mut bloom_filter = BlockedBloomFilter::new(1000, 0.01);
        bloom_filter.insert("test_key");
        bloom_filter.insert("another_key");

        // Test writing
        let bytes_written = writer
            .write(&bloom_filter)
            .expect("Failed to write bloom filter");

        // Verify file was created
        let path = temp_dir.path().join("bloom_filter");
        assert!(path.exists());

        // Verify header contents
        let file_path = format!("{}/bloom_filter", base_directory);
        let file = File::open(&file_path).expect("Failed to open persisted file");
        let mut reader = BufReader::new(file);

        // Read header fields
        let mut buf = [0u8; 8];
        assert!(reader.read_exact(&mut buf).is_ok());
        let block_size = u64::from_le_bytes(buf);

        assert!(reader.read_exact(&mut buf).is_ok());
        let num_hashes = u64::from_le_bytes(buf);

        assert!(reader.read_exact(&mut buf).is_ok());
        let bit_len = u64::from_le_bytes(buf);

        // Verify header values
        assert_eq!(block_size as usize, BlockedBloomFilter::BLOCK_SIZE_IN_BITS);
        assert_eq!(num_hashes as usize, bloom_filter.num_hash_functions());
        assert_eq!(bit_len as usize, bloom_filter.bits().len());

        // Verify bit data
        let expected_bits = bloom_filter.bits().as_raw_slice();
        let file_size = fs::metadata(&path).expect("Failed to get file size").len() as usize;
        let expected_file_size = 8 + 8 + 8 + expected_bits.len();
        assert_eq!(file_size, expected_file_size);
        assert_eq!(bytes_written, expected_file_size);

        // Test error cases
        let invalid_writer = BloomFilterWriter::new("/invalid/path".to_string());
        assert!(invalid_writer.write(&bloom_filter).is_err());
    }
}
