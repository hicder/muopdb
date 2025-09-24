use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{anyhow, Result};
use compression::compression::IntSeqEncoder;
use compression::elias_fano::ef::EliasFano;
use log::debug;
use utils::io::{append_file_to_writer, wrap_write};
use utils::on_disk_ordered_map::encoder::{IntegerCodec, VarintIntegerCodec};

use crate::terms::builder::{MultiTermBuilder, TermBuilder};

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

        // Remove term_map, offsets, posting lists files
        // std::fs::remove_file(term_map_path.as_str())?;
        // std::fs::remove_file(offset_path.as_str())?;
        // std::fs::remove_file(posting_list_path.as_str())?;

        Ok(())
    }
}

pub struct MultiTermWriter {
    base_dir: String,
}

impl MultiTermWriter {
    pub fn new(base_dir: String) -> Self {
        Self { base_dir }
    }

    pub fn write(&self, multi_builder: &mut MultiTermBuilder) -> Result<()> {
        for (user_id, builder) in multi_builder.builders_iter_mut() {
            let user_dir = Path::new(&self.base_dir).join(format!("user_{}", user_id));
            std::fs::create_dir_all(&user_dir)?;

            let writer = TermWriter::new(
                user_dir
                    .to_str()
                    .expect("user_dir should be valid UTF-8")
                    .to_string(),
            );
            writer.write(builder)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    #[test]
    fn test_term_writer() {
        let tmp_dir = tempdir::TempDir::new("test_term_writer").unwrap();
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
    fn test_multi_term_writer() {
        let tmp_dir = tempdir::TempDir::new("test_multi_term_writer").unwrap();
        let base_directory = tmp_dir.path().to_str().unwrap().to_string();

        let mut multi_builder = MultiTermBuilder::new(base_directory.clone());

        // Add terms for multiple users
        let user1 = 123u128;
        let user2 = 456u128;
        let user3 = 789u128;

        // User 1: Add some terms
        multi_builder
            .add(user1, 0, "apple:red".to_string())
            .unwrap();
        multi_builder
            .add(user1, 1, "banana:yellow".to_string())
            .unwrap();
        multi_builder
            .add(user1, 2, "apple:green".to_string())
            .unwrap();

        // User 2: Add different terms
        multi_builder
            .add(user2, 0, "car:toyota".to_string())
            .unwrap();
        multi_builder
            .add(user2, 1, "car:honda".to_string())
            .unwrap();
        multi_builder
            .add(user2, 2, "bike:yamaha".to_string())
            .unwrap();
        multi_builder
            .add(user2, 3, "car:toyota".to_string())
            .unwrap(); // Duplicate term

        // User 3: Add minimal terms
        multi_builder
            .add(user3, 0, "test:value".to_string())
            .unwrap();

        // Build all user builders
        multi_builder.build().unwrap();

        // Write using MultiTermWriter
        let multi_writer = MultiTermWriter::new(base_directory.clone());
        multi_writer.write(&mut multi_builder).unwrap();

        // Verify user directories were created
        let user1_dir = format!("{}/user_{}", base_directory, user1);
        let user2_dir = format!("{}/user_{}", base_directory, user2);
        let user3_dir = format!("{}/user_{}", base_directory, user3);

        assert!(fs::metadata(&user1_dir).unwrap().is_dir());
        assert!(fs::metadata(&user2_dir).unwrap().is_dir());
        assert!(fs::metadata(&user3_dir).unwrap().is_dir());

        // Verify combined files exist for each user
        let user1_combined = format!("{}/combined", user1_dir);
        let user2_combined = format!("{}/combined", user2_dir);
        let user3_combined = format!("{}/combined", user3_dir);

        assert!(fs::metadata(&user1_combined).unwrap().is_file());
        assert!(fs::metadata(&user2_combined).unwrap().is_file());
        assert!(fs::metadata(&user3_combined).unwrap().is_file());

        // Verify file sizes are reasonable (not empty)
        assert!(fs::metadata(&user1_combined).unwrap().len() > 0);
        assert!(fs::metadata(&user2_combined).unwrap().len() > 0);
        assert!(fs::metadata(&user3_combined).unwrap().len() > 0);

        // User 2 should have larger file (more terms)
        let user1_size = fs::metadata(&user1_combined).unwrap().len();
        let user2_size = fs::metadata(&user2_combined).unwrap().len();
        let user3_size = fs::metadata(&user3_combined).unwrap().len();

        assert!(user2_size >= user1_size); // User 2 has same or more terms
        assert!(user1_size > user3_size); // User 1 has more terms than user 3
    }

    #[test]
    fn test_multi_term_writer_empty_builder() {
        let tmp_dir = tempdir::TempDir::new("test_multi_term_writer_empty").unwrap();
        let base_directory = tmp_dir.path().to_str().unwrap().to_string();

        let mut multi_builder = MultiTermBuilder::new(base_directory.clone());

        // Build without adding any terms
        multi_builder.build().unwrap();

        // Write using MultiTermWriter
        let multi_writer = MultiTermWriter::new(base_directory.clone());
        multi_writer.write(&mut multi_builder).unwrap();

        // Should not create any user directories
        let entries = fs::read_dir(&base_directory).unwrap();
        let user_dirs: Vec<_> = entries
            .filter_map(|entry| {
                let entry = entry.ok()?;
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with("user_") {
                    Some(name)
                } else {
                    None
                }
            })
            .collect();

        assert!(
            user_dirs.is_empty(),
            "Should not create user directories for empty builder"
        );
    }

    #[test]
    fn test_multi_term_writer_single_user() {
        let tmp_dir = tempdir::TempDir::new("test_multi_term_writer_single").unwrap();
        let base_directory = tmp_dir.path().to_str().unwrap().to_string();

        let mut multi_builder = MultiTermBuilder::new(base_directory.clone());

        let user_id = 42u128;
        multi_builder
            .add(user_id, 0, "single:term".to_string())
            .unwrap();
        multi_builder
            .add(user_id, 1, "another:term".to_string())
            .unwrap();

        multi_builder.build().unwrap();

        let multi_writer = MultiTermWriter::new(base_directory.clone());
        multi_writer.write(&mut multi_builder).unwrap();

        // Should create exactly one user directory
        let user_dir = format!("{}/user_{}", base_directory, user_id);
        assert!(fs::metadata(&user_dir).unwrap().is_dir());

        // Combined file should exist
        let combined_path = format!("{}/combined", user_dir);
        assert!(fs::metadata(&combined_path).unwrap().is_file());
        assert!(fs::metadata(&combined_path).unwrap().len() > 0);
    }

    #[test]
    fn test_multi_term_writer_directory_creation() {
        let tmp_dir = tempdir::TempDir::new("test_multi_term_writer_dirs").unwrap();
        let base_directory = tmp_dir.path().to_str().unwrap().to_string();

        // Use nested directory structure
        let nested_base = format!("{}/nested/path", base_directory);
        fs::create_dir_all(&nested_base).unwrap();
        let mut multi_builder = MultiTermBuilder::new(nested_base.clone());

        let user_id = 999u128;
        multi_builder
            .add(user_id, 0, "nested:test".to_string())
            .unwrap();
        multi_builder.build().unwrap();

        let multi_writer = MultiTermWriter::new(nested_base.clone());
        multi_writer.write(&mut multi_builder).unwrap();

        // Should create nested directory structure
        let user_dir = format!("{}/user_{}", nested_base, user_id);
        assert!(fs::metadata(&user_dir).unwrap().is_dir());

        let combined_path = format!("{}/combined", user_dir);
        assert!(fs::metadata(&combined_path).unwrap().is_file());
    }

    #[test]
    fn test_multi_term_writer_error_propagation() {
        let tmp_dir = tempdir::TempDir::new("test_multi_term_writer_error").unwrap();
        let base_directory = tmp_dir.path().to_str().unwrap().to_string();

        let mut multi_builder = MultiTermBuilder::new(base_directory.clone());

        let user_id = 555u128;
        multi_builder
            .add(user_id, 0, "error:test".to_string())
            .unwrap();

        // Don't build - should cause error when writing
        let multi_writer = MultiTermWriter::new(base_directory.clone());
        let result = multi_writer.write(&mut multi_builder);

        // Should return error because builder wasn't built
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not built"));
    }

    #[test]
    fn test_multi_term_writer_user_isolation() {
        let tmp_dir = tempdir::TempDir::new("test_multi_term_writer_isolation").unwrap();
        let base_directory = tmp_dir.path().to_str().unwrap().to_string();

        let mut multi_builder = MultiTermBuilder::new(base_directory.clone());

        let user1 = 111u128;
        let user2 = 222u128;

        // Both users add the same term - should be isolated
        multi_builder
            .add(user1, 0, "shared:term".to_string())
            .unwrap();
        multi_builder
            .add(user1, 1, "user1:unique".to_string())
            .unwrap();

        multi_builder
            .add(user2, 0, "shared:term".to_string())
            .unwrap(); // Same term name
        multi_builder
            .add(user2, 1, "user2:unique".to_string())
            .unwrap();

        multi_builder.build().unwrap();

        let multi_writer = MultiTermWriter::new(base_directory.clone());
        multi_writer.write(&mut multi_builder).unwrap();

        // Both user directories should exist
        let user1_dir = format!("{}/user_{}", base_directory, user1);
        let user2_dir = format!("{}/user_{}", base_directory, user2);

        assert!(fs::metadata(&user1_dir).unwrap().is_dir());
        assert!(fs::metadata(&user2_dir).unwrap().is_dir());

        // Both should have their own combined files
        let user1_combined = format!("{}/combined", user1_dir);
        let user2_combined = format!("{}/combined", user2_dir);

        assert!(fs::metadata(&user1_combined).unwrap().is_file());
        assert!(fs::metadata(&user2_combined).unwrap().is_file());

        // Files should be roughly similar size (both have 2 terms)
        let user1_size = fs::metadata(&user1_combined).unwrap().len();
        let user2_size = fs::metadata(&user2_combined).unwrap().len();

        // Size difference should be 0
        assert!(
            user1_size == user2_size,
            "User file sizes too different: {} vs {}",
            user1_size,
            user2_size
        );
    }
}
