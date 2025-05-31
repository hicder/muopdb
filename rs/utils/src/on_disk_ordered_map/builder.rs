use std::collections::BTreeMap;
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};

use anyhow::Result;

use super::encoder::IntegerCodec;
use crate::io::append_file_to_writer;

const PAGE_SIZE: usize = 1024 * 1024; // 1 MB

fn shared_len_between_keys(key1: &[u8], key2: &[u8]) -> usize {
    key1.iter().zip(key2).take_while(|&(a, b)| a == b).count()
}

pub struct OnDiskOrderedMapBuilder {
    map: BTreeMap<String, u64>,
}

struct IndexItem {
    pub key: String,
    pub value: u64,
}

impl Default for OnDiskOrderedMapBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for the on disk ordered map. This will accumulate the keys and values in a BTreeMap.
/// Then on build, it will write the map to a file.
impl OnDiskOrderedMapBuilder {
    pub fn new() -> Self {
        OnDiskOrderedMapBuilder {
            map: BTreeMap::new(),
        }
    }

    /// Adds a key-value pair to the builder's in-memory map.
    ///
    /// If the key already exists, its value will be updated.
    /// The keys are stored in a BTreeMap, ensuring they are
    /// sorted before being written to disk during the build process.
    #[allow(dead_code)]
    pub fn add(&mut self, key: String, value: u64) {
        self.map.insert(key, value);
    }

    /// Adds a key-value pair to the builder's in-memory map.
    /// If the key already exists, returns its value.
    /// Otherwise, assigns a new value and returns it.
    pub fn add_or_get(&mut self, key: String, value: u64) -> u64 {
        if let Some(&value) = self.map.get(&key) {
            return value;
        }
        self.map.insert(key, value);
        value
    }

    /// Builds the on-disk ordered map by serializing the in-memory `BTreeMap` to a file.
    ///
    /// This process involves writing the key-value pairs to a data file and creating a separate
    /// index file for faster lookups. The data is written using key compression (shared prefixes)
    /// and values/lengths are encoded using the provided `IntegerCodec`. The file is structured
    /// into pages to potentially support partial loading.
    ///
    /// The final file structure is:
    /// 1. Codec identifier (4 bytes, little-endian)
    /// 2. Length of the index section (encoded using the codec)
    /// 3. Index data
    /// 4. Data section
    ///
    /// Temporary files (`data.bin` and `index.bin`) are created in a temporary directory
    /// within the parent directory of the target `file_path`, and then appended to the
    /// final file before the temporary directory is removed.
    ///
    /// # Arguments
    ///
    /// * `codec` - An implementation of the `IntegerCodec` trait to encode lengths and values.
    /// * `file_path` - The path where the final on-disk map file will be created.
    ///
    /// # Returns
    ///
    /// A `Result` indicating success (`Ok(())`) or an error if file operations or writing fails.
    ///
    /// # Errors
    ///
    /// This function can return an error if:
    /// - Creating directories or files fails.
    /// - Writing to files fails.
    /// - Flushing buffered writers fails.
    /// - Appending temporary files fails.
    /// - Removing the temporary directory fails.
    /// - The map is empty (causing `max().unwrap()` to panic - *Note: This should ideally be handled*).
    /// - File operations (open, write, close) fail.
    /// - I/O operations fail.
    ///
    /// Note: The current implementation handles some I/O errors via `Result`.
    ///
    pub fn build(&self, codec: impl IntegerCodec, file_path: &str) -> Result<()> {
        // Get parent of `file_path`
        let parent = std::path::Path::new(file_path).parent().unwrap();
        // Create a tmp dir inside parent
        std::fs::create_dir_all(parent.join("tmp"))?;
        let tmp_dir = parent.join("tmp");
        let data_path = tmp_dir.join("data.bin");

        // open the file for writing
        let mut data_file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(data_path)?;
        let mut data_buffered_writer = BufWriter::new(&mut data_file);

        let mut index_file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(tmp_dir.join("index.bin"))?;
        let mut index_buffered_writer = BufWriter::new(&mut index_file);

        let max_key_len = self.map.keys().map(|k| k.len()).max().unwrap();

        // write the map to the file
        // write the size of the key with varint first
        // avoid allocating a new buffer for each key
        let mut run_length = 0;
        let mut prev_key = Vec::<u8>::with_capacity(max_key_len);

        let mut indices = vec![];
        let mut total_written = 0;

        // Max size of varint is 9 bytes
        let mut buffer = vec![0u8; 9];

        // For each tuple, we will write shared, unshared, unshared key, value
        for (key, value) in self.map.iter() {
            if prev_key.is_empty() {
                indices.push(IndexItem {
                    key: key.clone(),
                    value: total_written as u64,
                });
            }
            let shared_len = shared_len_between_keys(&prev_key, key.as_bytes());
            // Write shared_len in varint
            buffer.fill(0);
            let encoded_len = codec.encode_u32(shared_len as u32, &mut buffer);
            data_buffered_writer
                .write_all(&buffer[..encoded_len])
                .unwrap();
            run_length += encoded_len;

            let unshared_len = key.len() - shared_len;

            // Write unshared_len in varint
            buffer.fill(0);
            let encoded_len = codec.encode_u32(unshared_len as u32, &mut buffer);
            data_buffered_writer
                .write_all(&buffer[..encoded_len])
                .unwrap();
            run_length += encoded_len;

            // Write the unshared part of the key
            data_buffered_writer
                .write_all(key.as_bytes()[shared_len..].as_ref())
                .unwrap();
            run_length += key.len() - shared_len;

            // Write value in varint
            buffer.fill(0);
            let encoded_len = codec.encode_u64(*value, &mut buffer);
            data_buffered_writer
                .write_all(&buffer[..encoded_len])
                .unwrap();
            run_length += encoded_len;

            // Add page break
            if run_length > PAGE_SIZE {
                total_written += run_length;
                run_length = 0;
                prev_key.clear();
            } else {
                prev_key.clear();
                prev_key.extend_from_slice(key.as_bytes());
            }
        }

        // Write index to file
        let mut index_len = 0u64;
        for index in indices {
            // Write key len first
            buffer.fill(0);
            let len = codec.encode_u32(index.key.len() as u32, &mut buffer);
            index_buffered_writer.write_all(&buffer[..len])?;
            index_buffered_writer.write_all(index.key.as_bytes())?;
            index_len += (len + index.key.len()) as u64;

            // Write value
            buffer.fill(0);
            let len = codec.encode_u64(index.value, &mut buffer);
            index_buffered_writer.write_all(&buffer[..len])?;
            index_len += len as u64;
        }

        data_buffered_writer.flush()?;
        index_buffered_writer.flush()?;

        let mut file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(file_path)?;

        let mut file_buffered_writer = BufWriter::new(&mut file);

        // write codec type
        file_buffered_writer.write_all(&codec.id().to_le_bytes())?;
        // write index length
        buffer.fill(0);
        let len = codec.encode_u64(index_len, &mut buffer);
        file_buffered_writer.write_all(&buffer[..len])?;

        // copy index data and append to file
        append_file_to_writer(
            tmp_dir.join("index.bin").to_str().unwrap(),
            &mut file_buffered_writer,
        )?;
        append_file_to_writer(
            tmp_dir.join("data.bin").to_str().unwrap(),
            &mut file_buffered_writer,
        )?;

        file_buffered_writer.flush()?;
        // rm tmp dir
        std::fs::remove_dir_all(tmp_dir)?;

        Ok(())
    }

    pub fn get_value(&self, key: &str) -> Option<u64> {
        self.map.get(key).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::on_disk_ordered_map::encoder::{FixedIntegerCodec, VarintIntegerCodec};

    #[test]
    fn test_map_builder() {
        let tmp_dir = tempdir::TempDir::new("test_builder").unwrap();
        let base_directory = tmp_dir.path().to_str().unwrap();
        let final_map_file_path = base_directory.to_string() + "/map.bin";

        let mut builder = OnDiskOrderedMapBuilder {
            map: BTreeMap::new(),
        };
        builder.add(String::from("key1"), 1);
        builder.add(String::from("key2"), 2);
        builder.add(String::from("key3"), 3);

        let codec = VarintIntegerCodec {};
        builder.build(codec, &final_map_file_path).unwrap();

        // Check that only map.bin is there
        assert_eq!(std::fs::read_dir(base_directory).unwrap().count(), 1);
        assert!(std::fs::read_dir(base_directory)
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path()
            .ends_with("map.bin"));
    }

    #[test]
    fn test_map_builder_fixed_integer_codec() {
        let tmp_dir = tempdir::TempDir::new("test_builder").unwrap();
        let base_directory = tmp_dir.path().to_str().unwrap();
        let final_map_file_path = base_directory.to_string() + "/map.bin";

        let mut builder = OnDiskOrderedMapBuilder {
            map: BTreeMap::new(),
        };
        builder.add(String::from("key1"), 1);
        builder.add(String::from("key2"), 2);
        builder.add(String::from("key3"), 3);

        let codec = FixedIntegerCodec {};
        builder.build(codec, &final_map_file_path).unwrap();

        // Check that only map.bin is there
        assert_eq!(std::fs::read_dir(base_directory).unwrap().count(), 1);
        assert!(std::fs::read_dir(base_directory)
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path()
            .ends_with("map.bin"));
    }
}
