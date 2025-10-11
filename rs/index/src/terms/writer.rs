use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{anyhow, Result};
use compression::compression::IntSeqEncoder;
use compression::elias_fano::ef::EliasFano;
use log::debug;
use odht::HashTableOwned;
use utils::io::{append_file_to_writer, wrap_write, write_pad};
use utils::on_disk_ordered_map::encoder::{IntegerCodec, VarintIntegerCodec};

use crate::terms::builder::{MultiTermBuilder, TermBuilder};
use crate::terms::term_index_info::{TermIndexInfo, TermIndexInfoHashTableConfig};

pub struct TermWriter {
    base_dir: String,
}

impl TermWriter {
    pub fn new(base_dir: String) -> Self {
        Self { base_dir }
    }

    /// Writes the term map, posting lists, and their offsets to disk.
    ///
    /// The data is organized into three main components:
    /// 1.  **Term Map**: A mapping from term IDs to term strings, stored as an `OnDiskOrderedMap`.
    /// 2.  **Posting Lists**: Elias-Fano encoded lists of document IDs for each term.
    /// 3.  **Offsets**: A list of offsets indicating the start position of each posting list within the posting lists file.
    ///
    /// These components are written into separate temporary files (`term_map`, `posting_lists`, `offsets`)
    /// and then combined into a single `combined` file in the following format:
    ///
    /// ```text
    /// +---------------------+
    /// | term_map_len (8B)   |
    /// +---------------------+
    /// | offsets_len (8B)    |
    /// +---------------------+
    /// | posting_lists_len (8B)|
    /// +---------------------+
    /// | Term Map Data       |
    /// | (padded to 8-byte   |
    /// |  alignment)         |
    /// +---------------------+
    /// | Offsets Data        |
    /// +---------------------+
    /// | Posting Lists Data  |
    /// +---------------------+
    /// ```
    ///
    /// # Arguments
    /// * `builder` - A mutable reference to a `TermBuilder` instance that has been built.
    ///
    /// # Returns
    /// `Result<()>` indicating success or an error if the `TermBuilder` is not built or an I/O error occurs.
    pub fn write(&self, builder: &mut TermBuilder) -> Result<()> {
        if !builder.is_built() {
            return Err(anyhow!("TermBuilder is not built"));
        }

        if builder.num_terms() == 0 {
            return Ok(());
        }

        // Write the term map
        let term_map_path = format!("{}/term_map", self.base_dir);
        builder
            .term_map
            .build(VarintIntegerCodec::new(), term_map_path.as_str())?;

        // Get length of term_map file
        let term_map_file = File::open(term_map_path.as_str()).unwrap();
        let term_map_len = term_map_file.metadata().unwrap().len();

        // Writer for posting lists
        let posting_list_path = format!("{}/posting_lists", self.base_dir);
        let mut pl_file = File::create(posting_list_path.as_str()).unwrap();
        let mut pl_writer = BufWriter::new(&mut pl_file);

        let num_terms = builder.num_terms();
        let mut offsets = Vec::with_capacity(num_terms as usize);
        let mut last_pl_offset = 0;
        for term_id in 0..num_terms {
            // elias fano encode
            let posting_list = builder.get_posting_list_by_id(term_id).unwrap();
            let mut encoder =
                EliasFano::new_encoder(*posting_list.last().unwrap(), posting_list.len());
            encoder.encode_batch(posting_list).unwrap();
            let len = encoder.write(&mut pl_writer)?;
            debug!(
                "[write] Term ID: {}, Offset: {}, Length: {}",
                term_id, last_pl_offset, len
            );
            offsets.push(last_pl_offset);
            last_pl_offset += len;
        }

        pl_writer.flush()?;

        // Write the offsets
        let offset_path = format!("{}/offsets", self.base_dir);
        let mut offset_file = File::create(offset_path.as_str()).unwrap();
        let mut offset_writer = BufWriter::new(&mut offset_file);
        let mut offset_len = 0;
        for offset in offsets {
            offset_len += wrap_write(&mut offset_writer, &offset.to_le_bytes())?;
        }
        offset_writer.flush()?;

        #[cfg(debug_assertions)]
        {
            // Open the file
            let pl_file = File::open(posting_list_path.as_str()).unwrap();
            let pl_file_len = pl_file.metadata().unwrap().len();
            assert_eq!(pl_file_len, last_pl_offset as u64);

            let offset_file = File::open(offset_path.as_str()).unwrap();
            let offset_file_len = offset_file.metadata().unwrap().len();
            assert_eq!(offset_file_len, offset_len as u64);
        }

        // Print the length for each file, in the same line
        debug!(
            "Term map length: {}, Offset length: {}, Posting list length: {}",
            term_map_len, offset_len, last_pl_offset
        );

        // Write the combined file. First, write the length of term_map file, then length of offsets file, then length of posting lists file
        let combined_path = format!("{}/combined", self.base_dir);
        let mut combined_file = File::create(combined_path.as_str()).unwrap();
        let mut combined_writer = BufWriter::new(&mut combined_file);
        wrap_write(&mut combined_writer, &term_map_len.to_le_bytes())?;
        wrap_write(&mut combined_writer, &offset_len.to_le_bytes())?;
        wrap_write(&mut combined_writer, &last_pl_offset.to_le_bytes())?;

        let mut total_size = 24;
        append_file_to_writer(term_map_path.as_str(), &mut combined_writer)?;
        total_size += term_map_len as usize;

        let padding = 8 - (total_size % 8);
        if padding != 8 {
            let padding_buffer = vec![0; padding];
            total_size += wrap_write(&mut combined_writer, &padding_buffer)?;
        }

        append_file_to_writer(offset_path.as_str(), &mut combined_writer)?;
        total_size += offset_len;
        let padding = 8 - (total_size % 8);
        if padding != 8 {
            let padding_buffer = vec![0; padding];
            total_size += wrap_write(&mut combined_writer, &padding_buffer)?;
        }

        append_file_to_writer(posting_list_path.as_str(), &mut combined_writer)?;
        total_size += last_pl_offset;
        combined_writer.flush()?;

        debug!("Total size: {}", total_size);

        // Remove term_map, offsets, posting lists files (okay to ignore errors)
        std::fs::remove_file(term_map_path.as_str()).ok();
        std::fs::remove_file(offset_path.as_str()).ok();
        std::fs::remove_file(posting_list_path.as_str()).ok();

        Ok(())
    }
}

pub struct MultiTermWriter {
    /// Base directory to write all user term indexes
    base_dir: String,
}

impl MultiTermWriter {
    pub fn new(base_dir: String) -> Self {
        Self { base_dir }
    }

    /// Combine all user term index files into one aligned global file
    /// and write the user term index info file.
    pub fn write(&self, builder: &MultiTermBuilder) -> Result<()> {
        if !builder.is_built() {
            return Err(anyhow!("MultiTermBuilder is not built"));
        }

        // Write each user's term index
        builder.for_each_builder_mut(|user_id, user_builder| {
            let user_dir = format!("{}/{}", self.base_dir, user_id);
            std::fs::create_dir_all(&user_dir).unwrap();
            let term_writer = TermWriter::new(user_dir);
            term_writer.write(user_builder).unwrap();
        });

        // Prepare output paths
        let combined_file_path = format!("{}/combined", self.base_dir);
        let user_index_info_file_path = format!("{}/user_term_index_info", self.base_dir);
        std::fs::create_dir_all(&self.base_dir)?;

        // Initialize writer and offset tracking
        let mut combined_file = File::create(&combined_file_path)?;
        let mut combined_file_writer = BufWriter::new(&mut combined_file);
        let mut total_written: usize = 0;
        let mut user_index_table = HashTableOwned::<TermIndexInfoHashTableConfig>::default();

        // Sort users for deterministic layout
        let mut user_ids = builder.get_user_ids();
        user_ids.sort();

        for user_id in user_ids {
            let user_dir = format!("{}/{}", self.base_dir, user_id);
            let user_file_path = format!("{}/combined", user_dir);

            // Ensure per-user combined file exists
            if !Path::new(&user_file_path).exists() {
                return Err(anyhow!(
                    "User {} combined file does not exist at {}",
                    user_id,
                    user_file_path
                ));
            }

            // Align to 8-byte boundary before writing this user’s data
            let padded = write_pad(total_written, &mut combined_file_writer, 8)?;
            total_written += padded;

            // Record offset
            let offset = total_written;

            // Append user’s combined term index file
            let length = append_file_to_writer(&user_file_path, &mut combined_file_writer)?;
            total_written += length;

            // Record metadata into hash table
            user_index_table.insert(
                &user_id,
                &TermIndexInfo {
                    offset: offset as u64,
                    length: length as u64,
                },
            );

            // Clean up per-user directory (ignore errors)
            std::fs::remove_dir_all(&user_dir).ok();
        }

        combined_file_writer.flush()?;

        // Write user index info to disk
        std::fs::write(&user_index_info_file_path, user_index_table.raw_bytes())?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::path::Path;

    use tempdir::TempDir;

    use super::*;
    use crate::terms::index::MultiTermIndex;

    #[test]
    fn test_term_writer() {
        let tmp_dir = TempDir::new("test_term_writer").unwrap();
        let base_directory = tmp_dir.path();

        let mut builder = TermBuilder::new(base_directory.join("scratch.tmp").as_path()).unwrap();
        for i in 0..10 {
            builder.add(i, format!("key{}", i % 3)).unwrap();
        }

        builder.build().unwrap();

        let base_dir_str = base_directory.to_str().unwrap().to_string();
        let writer = TermWriter::new(base_dir_str.clone());
        writer.write(&mut builder).unwrap();

        // Check the files
        let combined_path = format!("{}/combined", base_dir_str);
        let combined_file = File::open(combined_path.as_str()).unwrap();
        let combined_file_len = combined_file.metadata().unwrap().len();
        assert_eq!(combined_file_len, 216);
    }

    #[test]
    fn test_multi_term_writer_basic_roundtrip() {
        let tmp_dir = TempDir::new("test_multi_term_writer_roundtrip").unwrap();
        let base_dir = tmp_dir.path().to_str().unwrap().to_string();

        // Create multi-user builder
        let multi_builder = MultiTermBuilder::new(base_dir.clone());
        let user1 = 101u128;
        let user2 = 202u128;

        // Each user will have simple deterministic data
        // User 1: 2 terms
        multi_builder.add(user1, 0, "a".to_string()).unwrap();
        multi_builder.add(user1, 1, "b".to_string()).unwrap();

        // User 2: 1 term
        multi_builder.add(user2, 0, "x".to_string()).unwrap();

        // Build and write
        multi_builder.build().unwrap();
        let writer = MultiTermWriter::new(base_dir.clone());
        writer.write(&multi_builder).unwrap();

        // === File existence checks ===
        let combined_path = format!("{}/combined", base_dir);
        let combined_len = File::open(&combined_path)
            .unwrap()
            .metadata()
            .unwrap()
            .len();
        assert!(combined_len > 0, "combined file should not be empty");

        let info_path = format!("{}/user_term_index_info", base_dir);
        assert!(
            Path::new(&info_path).exists(),
            "user_term_index_info must exist"
        );

        // === Load MultiTermIndex ===
        let multi_index = MultiTermIndex::new(base_dir.clone()).unwrap();

        // Verify we can read terms for each user
        let term_id_a = multi_index.get_term_id_for_user(user1, "a").unwrap();
        let term_id_b = multi_index.get_term_id_for_user(user1, "b").unwrap();
        assert_ne!(term_id_a, term_id_b, "term IDs should differ");

        assert!(multi_index.get_term_id_for_user(user2, "x").is_ok());

        // === Validate offsets and lengths ===
        let user1_info = multi_index
            .term_index_info()
            .hash_table
            .get(&user1)
            .unwrap();
        let user2_info = multi_index
            .term_index_info()
            .hash_table
            .get(&user2)
            .unwrap();

        assert!(user1_info.length > 0);
        assert!(user2_info.length > 0);

        // User1 should start at offset 0
        assert_eq!(user1_info.offset, 0);

        // Offset for user2 must be 8-byte aligned
        assert_eq!(
            user2_info.offset % 8,
            0,
            "user2 offset must be 8-byte aligned"
        );

        // User2 should come immediately after user1 data (plus possible padding)
        let padding = (8 - (user1_info.length % 8)) % 8;
        assert_eq!(
            user2_info.offset,
            user1_info.length + padding,
            "user2 offset must match user1 length + padding"
        );

        // Combined file length consistency check
        let expected_total = user2_info.offset + user2_info.length;
        assert_eq!(
            combined_len, expected_total,
            "Combined file size should match sum of user sections"
        );
    }

    #[test]
    fn test_multi_term_writer_empty_builder() {
        let tmp_dir = TempDir::new("test_multi_term_writer_empty").unwrap();
        let base_dir = tmp_dir.path().to_str().unwrap().to_string();

        let multi_builder = MultiTermBuilder::new(base_dir.clone());
        multi_builder.build().unwrap();

        let writer = MultiTermWriter::new(base_dir.clone());
        writer.write(&multi_builder).unwrap();

        // Combined file should exist but be empty
        let combined_path = format!("{}/combined", base_dir);
        let combined_len = File::open(&combined_path)
            .unwrap()
            .metadata()
            .unwrap()
            .len();
        assert_eq!(combined_len, 0);

        // user_term_index_info should exist but be empty
        let user_index_info_path = format!("{}/user_term_index_info", base_dir);
        let info_bytes = std::fs::read(&user_index_info_path).unwrap();
        let hash_table =
            HashTableOwned::<TermIndexInfoHashTableConfig>::from_raw_bytes(&info_bytes)
                .expect("valid hash table");
        assert_eq!(hash_table.len(), 0);
    }

    #[test]
    fn test_multi_term_writer_error_on_unbuilt() {
        let tmp_dir = TempDir::new("test_multi_term_writer_error").unwrap();
        let base_dir = tmp_dir.path().to_str().unwrap().to_string();

        let multi_builder = MultiTermBuilder::new(base_dir.clone());
        multi_builder
            .add(1u128, 0, "term:fail".to_string())
            .unwrap();

        let writer = MultiTermWriter::new(base_dir.clone());
        let result = writer.write(&multi_builder);

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not built"));
    }
}
