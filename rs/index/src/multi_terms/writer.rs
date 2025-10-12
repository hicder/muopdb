use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::Path;

use anyhow::{anyhow, Result};
use odht::HashTableOwned;
use utils::io::{append_file_to_writer, write_pad};

use crate::multi_terms::builder::MultiTermBuilder;
use crate::terms::term_index_info::{TermIndexInfo, TermIndexInfoHashTableConfig};
use crate::terms::writer::TermWriter;

pub struct MultiTermWriter {
    /// Base directory to write all user term indexes
    base_dir: String,

    /// Segment directory
    segment_dir: Option<String>,
}

impl MultiTermWriter {
    pub fn new(base_dir: String) -> Self {
        Self {
            base_dir,
            segment_dir: None,
        }
    }

    pub fn new_with_segment_dir(segment_dir: String) -> Self {
        let base_dir = format!("{}/terms", segment_dir);
        Self {
            base_dir,
            segment_dir: Some(segment_dir),
        }
    }

    /// Combine all user term index files into one aligned global file
    /// and write the user term index info file.
    pub fn write(&self, builder: &MultiTermBuilder) -> Result<()> {
        if self.segment_dir.is_none() {
            self.write_with_reindex(builder, None)
        } else {
            let mappings = self.read_reassigned_mappings()?;
            self.write_with_reindex(builder, Some(&mappings))
        }
    }

    fn read_reassigned_mappings(&self) -> Result<HashMap<u128, Vec<i32>>> {
        // Scan all files in segment_dir with prefix "reassigned_mappings.". The suffix will be user_id
        let mut mappings = HashMap::new();

        // Check if segment_dir exists
        let segment_dir = self
            .segment_dir
            .as_ref()
            .ok_or_else(|| anyhow!("segment_dir is not set"))?;

        let mut files = std::fs::read_dir(segment_dir)
            .map_err(|e| anyhow!("Failed to read segment directory '{}': {}", segment_dir, e))?;

        while let Some(file_result) = files.next() {
            let file = file_result.map_err(|e| anyhow!("Failed to read directory entry: {}", e))?;

            let file_name = file
                .file_name()
                .into_string()
                .map_err(|_| anyhow!("Invalid file name (not valid UTF-8)"))?;

            if file_name.starts_with("reassigned_mappings.") {
                let user_id_str = file_name
                    .split('.')
                    .last()
                    .ok_or_else(|| anyhow!("Invalid file name format: {}", file_name))?;

                let user_id = user_id_str.parse::<u128>().map_err(|e| {
                    anyhow!(
                        "Failed to parse user ID from file name '{}': {}",
                        file_name,
                        e
                    )
                })?;

                let mut file = File::open(file.path())
                    .map_err(|e| anyhow!("Failed to open file '{}': {}", file_name, e))?;

                let mut user_mappings = Vec::new();

                // Read every 4 bytes, and parse as little endian
                let mut buffer = [0; 4];
                while file.read_exact(&mut buffer).is_ok() {
                    let idx = u32::from_le_bytes(buffer);
                    user_mappings.push(idx as i32);
                }

                mappings.insert(user_id, user_mappings);
            }
        }
        Ok(mappings)
    }

    /// Combine all user term index files into one aligned global file
    /// and write the user term index info file, with optional ID remapping for reindexing.
    ///
    /// # Arguments
    /// * `builder` - A built MultiTermBuilder
    /// * `id_mappings` - Optional mapping from user IDs to their respective ID mappings.
    ///                   If None, no remapping is applied.
    pub fn write_with_reindex(
        &self,
        builder: &MultiTermBuilder,
        id_mappings: Option<&std::collections::HashMap<u128, Vec<i32>>>,
    ) -> Result<()> {
        if !builder.is_built() {
            return Err(anyhow!("MultiTermBuilder is not built"));
        }

        // Write each user's term index
        builder.for_each_builder_mut(|user_id, user_builder| {
            let user_dir = format!("{}/{}", self.base_dir, user_id);
            std::fs::create_dir_all(&user_dir).unwrap();
            let term_writer = TermWriter::new(user_dir);

            // Get the ID mapping for this user if provided
            let id_mapping = id_mappings
                .and_then(|mappings| mappings.get(&user_id))
                .map(|mapping| mapping.as_slice());

            // Write with or without reindexing
            term_writer
                .write_with_reindex(user_builder, id_mapping)
                .unwrap();
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

            // Align to 8-byte boundary before writing this user's data
            let padded = write_pad(total_written, &mut combined_file_writer, 8)?;
            total_written += padded;

            // Record offset
            let offset = total_written;

            // Append user's combined term index file
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
    use crate::multi_terms::index::MultiTermIndex;

    #[test]
    fn test_multi_term_writer_basic_roundtrip() {
        let tmp_dir = TempDir::new("test_multi_term_writer_roundtrip").unwrap();
        let base_dir = tmp_dir.path().to_str().unwrap().to_string();

        // Create multi-user builder
        let multi_builder = MultiTermBuilder::new();
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

        let multi_builder = MultiTermBuilder::new();
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

        let multi_builder = MultiTermBuilder::new();
        multi_builder
            .add(1u128, 0, "term:fail".to_string())
            .unwrap();

        let writer = MultiTermWriter::new(base_dir.clone());
        let result = writer.write(&multi_builder);

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not built"));
    }

    #[test]
    fn test_multi_term_writer_with_reindex() {
        let tmp_dir = TempDir::new("test_multi_term_writer_reindex").unwrap();
        let base_dir = tmp_dir.path().to_str().unwrap().to_string();

        // Create multi-user builder
        let multi_builder = MultiTermBuilder::new();
        let user1 = 101u128;
        let user2 = 202u128;

        // User 1: 3 terms with point IDs 0, 1, 2
        multi_builder.add(user1, 0, "apple".to_string()).unwrap();
        multi_builder.add(user1, 1, "apple".to_string()).unwrap();
        multi_builder.add(user1, 2, "banana".to_string()).unwrap();

        // User 2: 2 terms with point IDs 0, 1
        multi_builder.add(user2, 0, "car".to_string()).unwrap();
        multi_builder.add(user2, 1, "car".to_string()).unwrap();

        // Build and write
        multi_builder.build().unwrap();

        // Create ID mappings for each user
        // User 1: Original 0,1,2 -> New 1,0,2 (reorder)
        // User 2: Original 0,1 -> New 0,1 (no change)
        let mut id_mappings = std::collections::HashMap::new();
        id_mappings.insert(user1, vec![1, 0, 2]);
        id_mappings.insert(user2, vec![0, 1]);

        let writer = MultiTermWriter::new(base_dir.clone());
        writer
            .write_with_reindex(&multi_builder, Some(&id_mappings))
            .unwrap();

        // Load MultiTermIndex and verify remapping worked
        let multi_index = MultiTermIndex::new(base_dir.clone()).unwrap();

        // Verify User 1's terms are remapped
        let apple_id = multi_index.get_term_id_for_user(user1, "apple").unwrap();
        let apple_pl: Vec<u32> = multi_index
            .get_or_create_index(user1)
            .unwrap()
            .get_posting_list_iterator(apple_id)
            .unwrap()
            .collect();
        // Original points 0,1 should now be 1,0 (and sorted)
        assert_eq!(apple_pl, vec![0, 1]);

        let banana_id = multi_index.get_term_id_for_user(user1, "banana").unwrap();
        let banana_pl: Vec<u32> = multi_index
            .get_or_create_index(user1)
            .unwrap()
            .get_posting_list_iterator(banana_id)
            .unwrap()
            .collect();
        // Original point 2 should still be 2
        assert_eq!(banana_pl, vec![2]);

        // Verify User 2's terms are unchanged
        let car_id = multi_index.get_term_id_for_user(user2, "car").unwrap();
        let car_pl: Vec<u32> = multi_index
            .get_or_create_index(user2)
            .unwrap()
            .get_posting_list_iterator(car_id)
            .unwrap()
            .collect();
        // Points should remain 0,1
        assert_eq!(car_pl, vec![0, 1]);
    }

    #[test]
    fn test_multi_term_writer_with_partial_reindex() {
        let tmp_dir = TempDir::new("test_multi_term_writer_partial_reindex").unwrap();
        let base_dir = tmp_dir.path().to_str().unwrap().to_string();

        // Create multi-user builder
        let multi_builder = MultiTermBuilder::new();
        let user1 = 101u128;
        let user2 = 202u128;

        // User 1: 3 terms with point IDs 0, 1, 2
        multi_builder.add(user1, 0, "term".to_string()).unwrap();
        multi_builder.add(user1, 1, "term".to_string()).unwrap();
        multi_builder.add(user1, 2, "term".to_string()).unwrap();

        // User 2: 2 terms with point IDs 0, 1
        multi_builder.add(user2, 0, "term".to_string()).unwrap();
        multi_builder.add(user2, 1, "term".to_string()).unwrap();

        // Build and write
        multi_builder.build().unwrap();

        // Create ID mapping only for User 1 (User 2 gets no remapping)
        // User 1: Original 0,1,2 -> New 0,2,1 (skip and reorder)
        let mut id_mappings = std::collections::HashMap::new();
        id_mappings.insert(user1, vec![0, 2, 1]);

        let writer = MultiTermWriter::new(base_dir.clone());
        writer
            .write_with_reindex(&multi_builder, Some(&id_mappings))
            .unwrap();

        // Load MultiTermIndex and verify partial remapping worked
        let multi_index = MultiTermIndex::new(base_dir.clone()).unwrap();

        // Verify User 1's terms are remapped
        let term_id = multi_index.get_term_id_for_user(user1, "term").unwrap();
        let term_pl: Vec<u32> = multi_index
            .get_or_create_index(user1)
            .unwrap()
            .get_posting_list_iterator(term_id)
            .unwrap()
            .collect();
        // Original points 0,1,2 should now be 0,2,1 (and sorted)
        assert_eq!(term_pl, vec![0, 1, 2]);

        // Verify User 2's terms are unchanged
        let term_id = multi_index.get_term_id_for_user(user2, "term").unwrap();
        let term_pl: Vec<u32> = multi_index
            .get_or_create_index(user2)
            .unwrap()
            .get_posting_list_iterator(term_id)
            .unwrap()
            .collect();
        // Points should remain 0,1
        assert_eq!(term_pl, vec![0, 1]);
    }

    #[test]
    fn test_multi_term_writer_with_reassigned_mappings() {
        let tmp_dir = TempDir::new("test_multi_term_writer_reassigned").unwrap();
        let segment_dir = tmp_dir.path().to_str().unwrap().to_string();

        // Create multi-user builder
        let multi_builder = MultiTermBuilder::new();
        let user1 = 101u128;
        let user2 = 202u128;

        // User 1: 3 terms with point IDs 0, 1, 2
        multi_builder.add(user1, 0, "apple".to_string()).unwrap();
        multi_builder.add(user1, 1, "banana".to_string()).unwrap();
        multi_builder.add(user1, 2, "cherry".to_string()).unwrap();

        // User 2: 2 terms with point IDs 0, 1
        multi_builder.add(user2, 0, "dog".to_string()).unwrap();
        multi_builder.add(user2, 1, "elephant".to_string()).unwrap();

        // Build and write
        multi_builder.build().unwrap();

        // Create reassigned mappings files for each user
        // User 1: Original 0,1,2 -> New 2,0,1 (reorder)
        // User 2: Original 0,1 -> New 1,0 (swap)

        // User 1 mappings file
        let user1_mappings_file = format!("{}/reassigned_mappings.{}", segment_dir, user1);
        std::fs::create_dir_all(&segment_dir).unwrap();
        let mut user1_file = File::create(user1_mappings_file).unwrap();
        user1_file.write_all(&2u32.to_le_bytes()).unwrap();
        user1_file.write_all(&0u32.to_le_bytes()).unwrap();
        user1_file.write_all(&1u32.to_le_bytes()).unwrap();

        // User 2 mappings file
        let user2_mappings_file = format!("{}/reassigned_mappings.{}", segment_dir, user2);
        let mut user2_file = File::create(user2_mappings_file).unwrap();
        user2_file.write_all(&1u32.to_le_bytes()).unwrap();
        user2_file.write_all(&0u32.to_le_bytes()).unwrap();

        // Create writer with segment_dir (will read reassigned mappings)
        let writer = MultiTermWriter::new_with_segment_dir(segment_dir.clone());
        writer.write(&multi_builder).unwrap();

        // Load MultiTermIndex and verify remapping worked
        let base_dir = format!("{}/terms", segment_dir);
        let multi_index = MultiTermIndex::new(base_dir.clone()).unwrap();

        // Verify User 1's terms are remapped according to reassigned mappings
        let apple_id = multi_index.get_term_id_for_user(user1, "apple").unwrap();
        let apple_pl: Vec<u32> = multi_index
            .get_or_create_index(user1)
            .unwrap()
            .get_posting_list_iterator(apple_id)
            .unwrap()
            .collect();
        // Original point 0 should now be 2 (and sorted)
        assert_eq!(apple_pl, vec![2]);

        let banana_id = multi_index.get_term_id_for_user(user1, "banana").unwrap();
        let banana_pl: Vec<u32> = multi_index
            .get_or_create_index(user1)
            .unwrap()
            .get_posting_list_iterator(banana_id)
            .unwrap()
            .collect();
        // Original point 1 should now be 0 (and sorted)
        assert_eq!(banana_pl, vec![0]);

        let cherry_id = multi_index.get_term_id_for_user(user1, "cherry").unwrap();
        let cherry_pl: Vec<u32> = multi_index
            .get_or_create_index(user1)
            .unwrap()
            .get_posting_list_iterator(cherry_id)
            .unwrap()
            .collect();
        // Original point 2 should now be 1 (and sorted)
        assert_eq!(cherry_pl, vec![1]);

        // Verify User 2's terms are remapped according to reassigned mappings
        let dog_id = multi_index.get_term_id_for_user(user2, "dog").unwrap();
        let dog_pl: Vec<u32> = multi_index
            .get_or_create_index(user2)
            .unwrap()
            .get_posting_list_iterator(dog_id)
            .unwrap()
            .collect();
        // Original point 0 should now be 1 (and sorted)
        assert_eq!(dog_pl, vec![1]);

        let elephant_id = multi_index.get_term_id_for_user(user2, "elephant").unwrap();
        let elephant_pl: Vec<u32> = multi_index
            .get_or_create_index(user2)
            .unwrap()
            .get_posting_list_iterator(elephant_id)
            .unwrap()
            .collect();
        // Original point 1 should now be 0 (and sorted)
        assert_eq!(elephant_pl, vec![0]);
    }
}
