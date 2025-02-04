use std::fs::{File, OpenOptions};
use std::io::Write;
use std::mem::size_of;

use anyhow::Result;

pub struct InvalidatedIdsStorage {
    base_directory: String,
    backing_file_size: usize,
    files: Vec<File>,
    current_backing_id: i32,
    current_offset: usize,
}

impl InvalidatedIdsStorage {
    pub fn new(base_directory: &str, backing_file_size: usize) -> Self {
        let bytes_per_invalidation = size_of::<u128>() + size_of::<u32>();
        let rounded_backing_file_size =
            backing_file_size / bytes_per_invalidation * bytes_per_invalidation;
        Self {
            base_directory: base_directory.to_string(),
            backing_file_size: rounded_backing_file_size,
            files: vec![],
            current_backing_id: -1,
            current_offset: rounded_backing_file_size,
        }
    }

    fn new_backing_file(&mut self) -> Result<()> {
        self.current_backing_id += 1;
        let backing_file = OpenOptions::new().create(true).append(true).open(format!(
            "{}/invalidated_ids.bin.{}",
            self.base_directory, self.current_backing_id
        ))?;

        self.files.push(backing_file);
        self.current_offset = 0;
        Ok(())
    }

    pub fn invalidate(&mut self, user_id: u128, point_id: u32) -> Result<()> {
        if self.current_offset == self.backing_file_size {
            self.new_backing_file()?;
        }

        let file = &mut self.files[self.current_backing_id as usize];
        file.write_all(&user_id.to_le_bytes())?;
        file.write_all(&point_id.to_le_bytes())?;
        self.current_offset += size_of::<u128>() + size_of::<u32>();

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::io::Read;
    use std::path::Path;

    use super::*;

    #[test]
    fn test_invalidate() {
        let temp_dir =
            tempdir::TempDir::new("test_invalidate").expect("Failed to create temporary directory");
        let base_dir = temp_dir
            .path()
            .to_str()
            .expect("Failed to convert temporary directory path to string")
            .to_string();
        let mut storage = InvalidatedIdsStorage::new(&base_dir, 1024);

        // Invalidate a user ID and point ID
        let user_id: u128 = 123456789012345678901234567890123456;
        let point_id: u32 = 987654321;
        assert!(storage.invalidate(user_id, point_id).is_ok());

        // Verify state after invalidation
        assert_eq!(storage.current_backing_id, 0);
        assert_eq!(storage.current_offset, size_of::<u128>() + size_of::<u32>());

        // Verify data written to the file
        let expected_file_path = format!("{}/invalidated_ids.bin.0", base_dir);
        let mut file = fs::File::open(expected_file_path).expect("Failed to open backing file");

        // Read the data back
        let mut buffer = vec![0u8; size_of::<u128>() + size_of::<u32>()];
        file.read_exact(&mut buffer)
            .expect("Failed to read from file");

        // Extract user_id and point_id from the buffer
        let user_id_bytes = &buffer[0..size_of::<u128>()];
        let point_id_bytes = &buffer[size_of::<u128>()..size_of::<u128>() + size_of::<u32>()];

        assert_eq!(user_id_bytes, &user_id.to_le_bytes());
        assert_eq!(point_id_bytes, &point_id.to_le_bytes());
    }

    #[test]
    fn test_invalidate_multiple_files() {
        let temp_dir = tempdir::TempDir::new("test_invalidate_multiple_files")
            .expect("Failed to create temporary directory");
        let base_dir = temp_dir
            .path()
            .to_str()
            .expect("Failed to convert temporary directory path to string")
            .to_string();
        let mut storage = InvalidatedIdsStorage::new(&base_dir, 64); // Small size for testing

        // Write multiple invalidations to trigger new backing files
        let num_invalidations = 10;
        let mut expected_data = Vec::new();
        for i in 0..num_invalidations {
            let user_id = i as u128;
            let point_id = i as u32;
            assert!(storage.invalidate(user_id, point_id).is_ok());
            expected_data.push((user_id, point_id));
        }

        // Verify that multiple backing files were created
        assert!(storage.current_backing_id > 0);

        // Verify contents of each file
        let bytes_per_entry = size_of::<u128>() + size_of::<u32>();
        let entries_per_file = storage.backing_file_size / bytes_per_entry;

        for backing_id in 0..=storage.current_backing_id {
            // Compute the range of entries expected in this file
            let start_index = backing_id as usize * entries_per_file;
            let end_index = ((backing_id as usize + 1) * entries_per_file).min(expected_data.len());
            let expected_entries = &expected_data[start_index..end_index];

            // Read the file
            let expected_file_path = format!("{}/invalidated_ids.bin.{}", base_dir, backing_id);
            assert!(Path::new(&expected_file_path).exists());

            let mut file_data = Vec::new();
            fs::File::open(&expected_file_path)
                .expect("Failed to open backing file")
                .read_to_end(&mut file_data)
                .expect("Failed to read from backing file");

            // Verify each entry in the file
            for (i, &(expected_user_id, expected_point_id)) in expected_entries.iter().enumerate() {
                let offset = i * bytes_per_entry;

                // Extract user_id and point_id from the file data
                let user_id_bytes = &file_data[offset..offset + size_of::<u128>()];
                let point_id_bytes =
                    &file_data[offset + size_of::<u128>()..offset + bytes_per_entry];

                let actual_user_id =
                    u128::from_le_bytes(user_id_bytes.try_into().expect("Invalid user ID bytes"));
                let actual_point_id =
                    u32::from_le_bytes(point_id_bytes.try_into().expect("Invalid point ID bytes"));

                // Compare with expected values
                assert_eq!(actual_user_id, expected_user_id);
                assert_eq!(actual_point_id, expected_point_id);
            }
        }

        // Ensure all entries were verified
        assert_eq!(num_invalidations, expected_data.len());
    }
}
