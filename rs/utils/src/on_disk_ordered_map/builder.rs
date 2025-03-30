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

/// Builder for the on disk ordered map. This will accumulate the keys and values in a BTreeMap.
/// Then on build, it will write the map to a file.
impl OnDiskOrderedMapBuilder {
    #[allow(dead_code)]
    fn add(&mut self, key: String, value: u64) {
        self.map.insert(key, value);
    }

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
        let mut prev_key = vec![0u8; max_key_len];

        let mut indices = vec![];
        let mut total_written = 0;

        // Max size of varint is 9 bytes
        let mut buffer = vec![0u8; 9];
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
        data_buffered_writer.flush()?;

        let mut file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(file_path)?;

        let mut file_buffered_writer = BufWriter::new(&mut file);

        // write codec type
        file_buffered_writer.write_all(&codec.id().to_le_bytes())?;
        // write index length
        file_buffered_writer.write_all(&index_len.to_le_bytes())?;
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::on_disk_ordered_map::encoder::VarintIntegerCodec;

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
}
