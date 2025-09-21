use std::cmp::max;
use std::fs::{create_dir_all, metadata, read_dir, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::mem::size_of;
use std::path::Path;

use anyhow::Result;

pub struct InvalidatedIdsStorage {
    base_directory: String,
    backing_file_size: usize,
    files: Vec<File>,
    current_backing_id: i32,
    current_offset: usize,
}

#[derive(PartialEq)]
pub struct InvalidatedUserDocId {
    pub user_id: u128,
    pub doc_id: u128,
}

pub struct InvalidatedIdsIterator {
    files: Vec<File>,
    current_file_idx: usize,
    current_offset: usize,
}

const BYTES_PER_INVALIDATION: usize = size_of::<u128>() + size_of::<u128>();

impl InvalidatedIdsStorage {
    const DEFAULT_BACKING_FILE_SIZE: usize = 8192;
    pub fn new(base_directory: &str, backing_file_size: usize) -> Self {
        let rounded_backing_file_size =
            backing_file_size / BYTES_PER_INVALIDATION * BYTES_PER_INVALIDATION;
        Self {
            base_directory: base_directory.to_string(),
            backing_file_size: rounded_backing_file_size,
            files: vec![],
            current_backing_id: -1,
            current_offset: rounded_backing_file_size,
        }
    }

    pub fn read(base_directory: &str) -> Result<Self> {
        let base_path = Path::new(base_directory);
        // Create a storage from scratch if none exists.
        if !base_path.exists() || !base_path.is_dir() {
            create_dir_all(base_directory)?;
            return Ok(Self::new(base_directory, Self::DEFAULT_BACKING_FILE_SIZE));
        }

        // Get the list of invalidated ids files
        let mut invalidated_ids_files: Vec<String> = read_dir(base_directory)?
            .filter_map(|entry| entry.ok()) // Ignore errors
            .filter_map(|entry| {
                let file_name = entry.file_name();
                let file_name_str = file_name.to_str()?.to_string();

                if file_name_str.starts_with("invalidated_ids.bin.") {
                    Some(entry.path().to_str()?.to_string())
                } else {
                    None
                }
            })
            .collect();

        if invalidated_ids_files.is_empty() {
            return Ok(Self::new(base_directory, Self::DEFAULT_BACKING_FILE_SIZE));
        }

        // Sort the files numerically by their suffix
        invalidated_ids_files.sort_by_key(|file_name| {
            file_name
                .rsplit_once('.') // Split at the last dot
                .and_then(|(_, suffix)| suffix.parse::<u32>().ok()) // Parse the suffix as a number
                .unwrap_or(0) // Default to 0 if parsing fails
        });
        let last_file = invalidated_ids_files.last().unwrap();
        let current_offset = metadata(last_file)?.len() as usize;

        // - If there are multiple files, all files (except maybe last one) should have the same
        // size, and backing file size should be this
        // - If there is only one file, backing file size should be max between this file size and
        // default file size
        let first_file_size = metadata(invalidated_ids_files.first().unwrap())?.len() as usize;
        let backing_file_size = if invalidated_ids_files.len() == 1 {
            max(Self::DEFAULT_BACKING_FILE_SIZE, first_file_size)
        } else {
            first_file_size
        };

        let files: Vec<File> = invalidated_ids_files
            .iter()
            .filter_map(|file_path| OpenOptions::new().append(true).open(file_path).ok())
            .collect();

        let rounded_backing_file_size =
            backing_file_size / BYTES_PER_INVALIDATION * BYTES_PER_INVALIDATION;
        Ok(Self {
            base_directory: base_directory.to_string(),
            backing_file_size: rounded_backing_file_size,
            files,
            current_backing_id: invalidated_ids_files.len() as i32 - 1,
            current_offset,
        })
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

    pub fn invalidate(&mut self, user_id: u128, doc_id: u128) -> Result<()> {
        if self.current_offset == self.backing_file_size {
            self.new_backing_file()?;
        }

        let file = &mut self.files[self.current_backing_id as usize];

        let mut buffer = Vec::with_capacity(BYTES_PER_INVALIDATION);

        // Write user_id and doc_id into the buffer
        buffer.extend_from_slice(&user_id.to_le_bytes());
        buffer.extend_from_slice(&doc_id.to_le_bytes());

        // Perform a single atomic write
        file.write_all(&buffer)?;

        // Ensure all written data is flushed to disk
        file.sync_all()?;
        self.current_offset += BYTES_PER_INVALIDATION;

        Ok(())
    }

    /// Batch invalidates multiple `InvalidatedUserDocId`s without fsyncing on every invalidation.
    /// Performs fsync at the end of each file.
    pub fn invalidate_batch(&mut self, invalidations: &[InvalidatedUserDocId]) -> Result<()> {
        let mut buffer = Vec::with_capacity(self.backing_file_size);

        for invalidation in invalidations {
            // Check if the current file has enough space for this invalidation
            if self.current_offset == self.backing_file_size {
                if !buffer.is_empty() {
                    let file = &mut self.files[self.current_backing_id as usize];
                    file.write_all(&buffer)?;
                    file.sync_all()?;

                    // Clear the buffer and create a new backing file
                    buffer.clear();
                }
                self.new_backing_file()?;
            }

            // Write user_id and doc_id into the buffer
            buffer.extend_from_slice(&invalidation.user_id.to_le_bytes());
            buffer.extend_from_slice(&invalidation.doc_id.to_le_bytes());

            // Update the current offset after writing to the buffer
            self.current_offset += BYTES_PER_INVALIDATION;
        }

        // Write and sync any remaining data in the buffer to the current file
        if !buffer.is_empty() {
            let file = &mut self.files[self.current_backing_id as usize];
            file.write_all(&buffer)?;
            file.sync_all()?;
        }

        Ok(())
    }

    pub fn iter(&self) -> InvalidatedIdsIterator {
        InvalidatedIdsIterator {
            files: (0..self.files.len())
                .filter_map(|i| {
                    OpenOptions::new()
                        .read(true)
                        .open(format!("{}/invalidated_ids.bin.{}", self.base_directory, i))
                        .ok()
                })
                .collect(),
            current_file_idx: 0,
            current_offset: 0,
        }
    }

    pub fn base_directory(&self) -> &str {
        &self.base_directory
    }

    pub fn num_entries(&self) -> usize {
        if self.current_backing_id == -1 {
            0
        } else {
            (self.current_offset + self.current_backing_id as usize * self.backing_file_size)
                / BYTES_PER_INVALIDATION
        }
    }
}

impl Iterator for InvalidatedIdsIterator {
    type Item = InvalidatedUserDocId;

    fn next(&mut self) -> Option<Self::Item> {
        while self.current_file_idx < self.files.len() {
            let file = &mut self.files[self.current_file_idx];

            // Get the current file size
            let file_size = file.metadata().unwrap().len() as usize;

            if self.current_offset >= file_size {
                // Move to the next file if we've reached the end of the current file
                self.current_file_idx += 1;
                self.current_offset = 0;
                continue;
            }

            if self.current_offset > file_size - BYTES_PER_INVALIDATION {
                panic!("Incomplete invalidation record at end of file");
            }

            file.seek(SeekFrom::Start(self.current_offset as u64))
                .unwrap();

            let mut buffer = [0u8; BYTES_PER_INVALIDATION];
            // FIXME: this incurs one syscall per iteration. but hopefully this file is small and
            // we only do it at the start. mmap will probably help here. Update is necessary.
            file.read_exact(&mut buffer).unwrap();

            let user_id = u128::from_le_bytes(buffer[..16].try_into().unwrap());
            let doc_id = u128::from_le_bytes(buffer[16..].try_into().unwrap());

            self.current_offset += BYTES_PER_INVALIDATION;

            return Some(InvalidatedUserDocId { user_id, doc_id });
        }
        None
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

        // Invalidate a user ID and doc ID
        let user_id: u128 = 123456789012345678901234567890123456;
        let doc_id: u128 = 987654321;
        assert!(storage.invalidate(user_id, doc_id).is_ok());

        // Verify state after invalidation
        assert_eq!(storage.current_backing_id, 0);
        assert_eq!(
            storage.current_offset,
            size_of::<u128>() + size_of::<u128>()
        );
        assert_eq!(storage.num_entries(), 1);

        // Verify data written to the file
        let expected_file_path = format!("{}/invalidated_ids.bin.0", base_dir);
        let mut file = fs::File::open(expected_file_path).expect("Failed to open backing file");

        // Read the data back
        let mut buffer = vec![0u8; size_of::<u128>() + size_of::<u128>()];
        file.read_exact(&mut buffer)
            .expect("Failed to read from file");

        // Extract user_id and doc_id from the buffer
        let user_id_bytes = &buffer[0..size_of::<u128>()];
        let doc_id_bytes = &buffer[size_of::<u128>()..size_of::<u128>() + size_of::<u128>()];

        assert_eq!(user_id_bytes, &user_id.to_le_bytes());
        assert_eq!(doc_id_bytes, &doc_id.to_le_bytes());
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
            let doc_id = i as u128;
            assert!(storage.invalidate(user_id, doc_id).is_ok());
            expected_data.push((user_id, doc_id));
        }

        // Verify that multiple backing files were created
        assert!(storage.current_backing_id > 0);
        assert_eq!(storage.num_entries(), 10);

        // Verify contents of each file
        let bytes_per_entry = size_of::<u128>() + size_of::<u128>();
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
            for (i, &(expected_user_id, expected_doc_id)) in expected_entries.iter().enumerate() {
                let offset = i * bytes_per_entry;

                // Extract user_id and doc_id from the file data
                let user_id_bytes = &file_data[offset..offset + size_of::<u128>()];
                let doc_id_bytes = &file_data[offset + size_of::<u128>()..offset + bytes_per_entry];

                let actual_user_id =
                    u128::from_le_bytes(user_id_bytes.try_into().expect("Invalid user ID bytes"));
                let actual_doc_id =
                    u128::from_le_bytes(doc_id_bytes.try_into().expect("Invalid doc ID bytes"));

                // Compare with expected values
                assert_eq!(actual_user_id, expected_user_id);
                assert_eq!(actual_doc_id, expected_doc_id);
            }
        }

        // Ensure all entries were verified
        assert_eq!(num_invalidations, expected_data.len());
    }

    #[test]
    fn test_read_single_file() {
        let temp_dir = tempdir::TempDir::new("test_read_single_file")
            .expect("Failed to create temporary directory");
        let base_dir = temp_dir
            .path()
            .to_str()
            .expect("Failed to convert temporary directory path to string")
            .to_string();
        let mut storage = InvalidatedIdsStorage::new(&base_dir, 1024);

        // Invalidate a user ID and doc ID
        let user_id: u128 = 123456789012345678901234567890123456;
        let doc_id: u128 = 987654321;
        assert!(storage.invalidate(user_id, doc_id).is_ok());

        let mut read_back_storage = InvalidatedIdsStorage::read(&base_dir)
            .expect("Failed to read back invalidated ids storage");

        // Verify state after invalidation
        assert_eq!(read_back_storage.current_backing_id, 0);
        assert_eq!(
            read_back_storage.current_offset,
            size_of::<u128>() + size_of::<u128>()
        );

        assert_eq!(
            read_back_storage.backing_file_size,
            InvalidatedIdsStorage::DEFAULT_BACKING_FILE_SIZE / BYTES_PER_INVALIDATION
                * BYTES_PER_INVALIDATION
        );

        // Test iterator
        let invalidated_id = read_back_storage.iter().next();
        assert!(invalidated_id.as_ref().is_some());
        assert_eq!(invalidated_id.as_ref().unwrap().user_id, user_id);
        assert_eq!(invalidated_id.as_ref().unwrap().doc_id, doc_id);

        // Test invalidating read back storage
        assert!(read_back_storage.invalidate(user_id, doc_id + 1).is_ok());
        assert_eq!(
            storage.current_offset + size_of::<u128>() + size_of::<u128>(),
            read_back_storage.current_offset
        );
    }

    #[test]
    fn test_read_multiple_files() {
        let temp_dir = tempdir::TempDir::new("test_read_multiple_files")
            .expect("Failed to create temporary directory");
        let base_dir = temp_dir
            .path()
            .to_str()
            .expect("Failed to convert temporary directory path to string")
            .to_string();
        let mut storage = InvalidatedIdsStorage::new(&base_dir, 64); // Small size for testing

        // Write multiple invalidations to trigger new backing files
        let num_invalidations = 31;
        let mut expected_data = Vec::new();
        for i in 0..num_invalidations {
            let user_id = i as u128;
            let doc_id = i as u128;
            assert!(storage.invalidate(user_id, doc_id).is_ok());
            expected_data.push((user_id, doc_id));
        }

        let mut read_back_storage = InvalidatedIdsStorage::read(&base_dir)
            .expect("Failed to read back invalidated ids storage");

        assert_eq!(
            storage.current_backing_id,
            read_back_storage.current_backing_id
        );
        assert_eq!(storage.current_offset, read_back_storage.current_offset);
        assert_eq!(
            storage.backing_file_size,
            read_back_storage.backing_file_size
        );

        // Test iterator
        let iter_data: Vec<InvalidatedUserDocId> = read_back_storage.iter().collect();
        assert_eq!(iter_data.len(), expected_data.len());
        for (expected, actual) in expected_data.iter().zip(iter_data.iter()) {
            assert_eq!(expected.0, actual.user_id);
            assert_eq!(expected.1, actual.doc_id);
        }

        // Test invalidating read back storage
        assert!(read_back_storage.invalidate(31, 31).is_ok());
        assert_eq!(
            storage.current_offset + size_of::<u128>() + size_of::<u128>(),
            read_back_storage.current_offset
        );

        // Test iterator again after adding a new item
        let iter_data_after_add: Vec<InvalidatedUserDocId> = read_back_storage.iter().collect();
        assert_eq!(iter_data_after_add.len(), expected_data.len() + 1);
        assert_eq!(iter_data_after_add.last().unwrap().user_id, 31);
        assert_eq!(iter_data_after_add.last().unwrap().doc_id, 31);
    }

    #[test]
    fn test_invalidate_batch() {
        let temp_dir = tempdir::TempDir::new("test_invalidate_batch")
            .expect("Failed to create temporary directory");
        let base_dir = temp_dir
            .path()
            .to_str()
            .expect("Failed to convert temporary directory path to string")
            .to_string();
        let mut storage = InvalidatedIdsStorage::new(&base_dir, 1024);

        // Case 1: only fill the current backing file
        let user_id1: u128 = 123456789012345678901234567890123456;
        let doc_id1: u128 = 987654321;
        let user_id2: u128 = 223456789012345678901234567890123456;
        let doc_id2: u128 = 987654322;

        assert!(storage
            .invalidate_batch(&[
                InvalidatedUserDocId {
                    user_id: user_id1,
                    doc_id: doc_id1
                },
                InvalidatedUserDocId {
                    user_id: user_id2,
                    doc_id: doc_id2
                }
            ])
            .is_ok());

        // Verify state after case 1
        let mut read_back_storage = InvalidatedIdsStorage::read(&base_dir)
            .expect("Failed to read back invalidated ids storage");
        assert_eq!(read_back_storage.current_backing_id, 0);
        assert_eq!(
            read_back_storage.current_offset,
            2 * (size_of::<u128>() + size_of::<u128>())
        );

        // Verify invalidations in case 1
        let mut iter = read_back_storage.iter();
        let invalidation1 = iter.next();
        assert!(invalidation1.is_some());
        assert_eq!(invalidation1.as_ref().unwrap().user_id, user_id1);
        assert_eq!(invalidation1.as_ref().unwrap().doc_id, doc_id1);

        let invalidation2 = iter.next();
        assert!(invalidation2.is_some());
        assert_eq!(invalidation2.as_ref().unwrap().user_id, user_id2);
        assert_eq!(invalidation2.as_ref().unwrap().doc_id, doc_id2);

        // Case 2: write full buffers to multiple files
        let large_invalidations: Vec<InvalidatedUserDocId> = (0..2048)
            .map(|i| InvalidatedUserDocId {
                user_id: user_id1 + i as u128,
                doc_id: doc_id1 + i as u128,
            })
            .collect();
        assert!(storage.invalidate_batch(&large_invalidations).is_ok());

        // Verify state after case 2
        read_back_storage = InvalidatedIdsStorage::read(&base_dir)
            .expect("Failed to read back invalidated ids storage");
        assert_eq!(
            read_back_storage.current_offset,
            2 * (size_of::<u128>() + size_of::<u128>())
        );

        // Verify invalidation in case 2
        iter = read_back_storage.iter();

        // Skip previous invalidations
        for _ in 0..2 {
            iter.next();
        }

        for large_invalidation in large_invalidations.iter() {
            let invalidation = iter.next();
            assert!(invalidation.is_some());
            assert_eq!(
                invalidation.as_ref().unwrap().user_id,
                large_invalidation.user_id
            );
            assert_eq!(
                invalidation.as_ref().unwrap().doc_id,
                large_invalidation.doc_id
            );
        }
    }
}
