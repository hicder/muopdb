use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::vec;

use anyhow::{anyhow, Result};
use utils::io::wrap_write;
use utils::mem::transmute_slice_to_u8;

use super::{PostingList, PostingListStorage, PostingListStorageConfig};

const PL_METADATA_LEN: usize = 2;

#[derive(Debug)]
struct FileAccessInfo {
    file_num: usize,
    file_offset: usize,
}

pub struct FileBackedAppendablePostingListStorage {
    pub memory_threshold: usize,
    pub backing_file_size: usize,
    // Number of clusters in the IVF. Each cluster should have one posting list.
    num_clusters: usize,
    // Counter of appended posting lists.
    current_num_of_posting_list: usize,
    // Number of bytes required to store the data.
    // - If it's in memory, the data stored are the posting lists
    // - If it's on disk, the data stored are the posting lists + metadata
    size_bytes: usize,

    // Whether it's currently in memory
    resident_posting_lists: Vec<Vec<u64>>,
    resident: bool,

    // Only has value if we spill to disk
    base_directory: String,
    mmaps: Vec<memmap2::MmapMut>,
    current_backing_id: i32,
    current_offset: usize,
    offset_to_current_posting_list: u64,
}

impl FileBackedAppendablePostingListStorage {
    pub fn new(
        base_directory: String,
        memory_threshold: usize,
        backing_file_size: usize,
        num_clusters: usize,
    ) -> Self {
        let pl_metadata_in_bytes = PL_METADATA_LEN * std::mem::size_of::<u64>();
        let offset_to_current_posting_list = (num_clusters * pl_metadata_in_bytes) as u64;
        // Rounding to PL_METADATA_LEN * `u64` size in bytes to at least simplify the reading of
        // posting list offsets and lengths.
        //
        // That's the best we can do since we do not know sizes of posting lists
        // beforehand.
        let rounded_backing_file_size =
            backing_file_size / pl_metadata_in_bytes * pl_metadata_in_bytes;
        Self {
            base_directory,
            memory_threshold,
            backing_file_size: rounded_backing_file_size,
            num_clusters,
            current_num_of_posting_list: 0,
            resident_posting_lists: vec![],
            resident: true,
            size_bytes: 0,
            mmaps: vec![],
            current_backing_id: -1,
            current_offset: 0,
            offset_to_current_posting_list,
        }
    }

    pub fn new_with_config(base_directory: String, config: PostingListStorageConfig) -> Self {
        Self::new(
            base_directory,
            config.memory_threshold,
            config.file_size,
            config.num_clusters,
        )
    }

    pub fn is_resident(&self) -> bool {
        self.resident
    }

    pub fn flush(&mut self) -> Result<()> {
        if !self.resident {
            // Flush all mmaps
            for mmap in self.mmaps.iter_mut() {
                mmap.flush()?;
            }
        }
        Ok(())
    }

    fn new_backing_file(&mut self) -> Result<()> {
        self.current_backing_id += 1;
        let backing_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(format!(
                "{}/posting_list.bin.{}",
                self.base_directory, self.current_backing_id
            ))?;

        backing_file.set_len(self.backing_file_size as u64)?;

        self.mmaps
            .push(unsafe { memmap2::MmapMut::map_mut(&backing_file)? });
        self.current_offset = 0;
        Ok(())
    }

    fn write_to_current_mmap(&mut self, data: &[u8]) -> Result<()> {
        let mmap = &mut self.mmaps[self.current_backing_id as usize];
        let write_size = data.len();

        mmap[self.current_offset..self.current_offset + write_size].copy_from_slice(data);
        self.current_offset += write_size;

        Ok(())
    }

    fn get_posting_list_offset(&self) -> usize {
        let pl_metadata_in_bytes = PL_METADATA_LEN * std::mem::size_of::<u64>();
        self.num_clusters * pl_metadata_in_bytes
    }

    fn flush_resident_posting_lists_to_disk(&mut self) -> Result<()> {
        if !self.resident {
            return Err(anyhow!(
                "Posting lists should still be in memory when flushing to disk"
            ));
        }
        // Trigger the creation of a new file when we start to flush
        self.current_offset = self.backing_file_size;
        let size_in_bytes = std::mem::size_of::<u64>();
        // Extract all the data we need from self to avoid immutable borrow issue
        let posting_lists: Vec<Vec<u64>> = std::mem::take(&mut self.resident_posting_lists);
        // First write the metadata
        for posting_list in &posting_lists {
            if self.current_offset == self.backing_file_size {
                self.new_backing_file()?;
            }
            // Write the length of the posting list
            self.write_to_current_mmap(&posting_list.len().to_le_bytes())?;

            // Write the offset to the current posting list
            self.write_to_current_mmap(&self.offset_to_current_posting_list.to_le_bytes())?;
            self.offset_to_current_posting_list += (posting_list.len() * size_in_bytes) as u64;
        }
        // Now write the posting lists
        let metadata_tot_size = self.get_posting_list_offset();
        let file_num_for_first_posting_list = metadata_tot_size / self.backing_file_size;
        // file_num_for_first_posting_list is an index, while self.mmaps.len() is the length
        while self.mmaps.len() - 1 < file_num_for_first_posting_list {
            self.new_backing_file()?;
        }
        let file_access_info = self.offset_to_file_access_info(self.get_posting_list_offset())?;
        self.current_offset = file_access_info.file_offset;
        self.current_backing_id = file_access_info.file_num as i32;

        for posting_list in &posting_lists {
            for idx in posting_list {
                if self.current_offset == self.backing_file_size {
                    self.new_backing_file()?;
                }
                self.write_to_current_mmap(&idx.to_le_bytes())?;
            }
        }

        self.resident = false;
        self.resident_posting_lists.clear();
        Ok(())
    }

    // The caller is responsible for setting self.current_offset to the right position
    // for writing metadata.
    fn append_posting_list_to_disk(&mut self, posting_list: &[u64]) -> Result<()> {
        if self.resident {
            return Err(anyhow!("Posting lists should already be flushed to disk"));
        }
        // First write the metadata
        if self.current_offset == self.backing_file_size {
            self.new_backing_file()?;
        }
        // Write the length of the posting list
        self.write_to_current_mmap(&posting_list.len().to_le_bytes())?;

        // Write the offset to the current posting list
        self.write_to_current_mmap(&self.offset_to_current_posting_list.to_le_bytes())?;

        // Now write the posting list
        let size_in_bytes = std::mem::size_of::<u64>();
        let metadata_tot_size = self.get_posting_list_offset();
        let file_num_for_first_posting_list = metadata_tot_size / self.backing_file_size;
        // file_num_for_first_posting_list is an index, while self.mmaps.len() is the length
        while self.mmaps.len() - 1 < file_num_for_first_posting_list {
            self.new_backing_file()?;
        }
        let file_access_info =
            self.offset_to_file_access_info(self.offset_to_current_posting_list as usize)?;
        self.current_offset = file_access_info.file_offset;
        self.current_backing_id = file_access_info.file_num as i32;

        self.offset_to_current_posting_list += (posting_list.len() * size_in_bytes) as u64;

        for idx in posting_list.iter() {
            if self.current_offset == self.backing_file_size {
                self.new_backing_file()?;
            }
            self.write_to_current_mmap(&idx.to_le_bytes())?;
        }
        self.current_num_of_posting_list += 1;
        Ok(())
    }

    fn offset_to_file_access_info(&self, offset: usize) -> Result<FileAccessInfo> {
        let file_num = offset / self.backing_file_size;
        if file_num >= self.mmaps.len() {
            return Err(anyhow!("File number out of bound"));
        }

        let file_offset = offset % self.backing_file_size;
        if file_offset >= self.mmaps[file_num].len() {
            return Err(anyhow!("File offset out of bound"));
        }

        Ok(FileAccessInfo {
            file_num,
            file_offset,
        })
    }
}

impl<'a> PostingListStorage<'a> for FileBackedAppendablePostingListStorage {
    fn get(&'a self, id: u32) -> Result<PostingList<'a>> {
        let i = id as usize;

        if self.resident {
            if i >= self.resident_posting_lists.len() {
                return Err(anyhow!("Posting list id out of bound"));
            }
            return Ok(PostingList::new_with_slices(vec![transmute_slice_to_u8(
                &self.resident_posting_lists[i],
            )]));
        }

        if i >= self.current_num_of_posting_list {
            return Err(anyhow!("Posting list id out of bound"));
        }
        let size_in_bytes = std::mem::size_of::<u64>();
        let offset_to_pl_metadata = i * PL_METADATA_LEN * size_in_bytes;

        let file_access_info = self.offset_to_file_access_info(offset_to_pl_metadata)?;
        let file_offset = file_access_info.file_offset as usize;
        let file_num = file_access_info.file_num as usize;
        let mmap = &self.mmaps[file_num];
        let slice = &mmap[file_offset..file_offset + size_in_bytes];
        let pl_len = u64::from_le_bytes(slice.try_into()?) as usize;
        let slice =
            &mmap[file_offset + size_in_bytes..file_offset + PL_METADATA_LEN * size_in_bytes];
        let pl_offset = u64::from_le_bytes(slice.try_into()?) as usize;

        let file_access_info = self.offset_to_file_access_info(pl_offset)?;
        let required_size = pl_len * size_in_bytes;

        // Posting list fits within a single mmap
        let file_offset = file_access_info.file_offset as usize;
        let file_num = file_access_info.file_num as usize;
        if file_offset + required_size <= mmap.len() {
            let mmap = &self.mmaps[file_num as usize];
            let slice = &mmap[file_offset..file_offset + required_size];
            return Ok(PostingList::new_with_slices(vec![slice]));
        }

        // Posting list spans across multiple mmaps.
        let mut posting_list = PostingList::new();
        let mut remaining_elem = pl_len;
        let mut current_file_num = file_num;
        let mut current_offset = file_offset;
        while remaining_elem > 0 {
            let mmap = &self.mmaps[current_file_num];
            let bytes_left_in_mmap = mmap.len() - current_offset;
            let elems_in_mmap = std::cmp::min(remaining_elem, bytes_left_in_mmap / size_in_bytes);

            let slice = &mmap[current_offset..current_offset + elems_in_mmap * size_in_bytes];
            posting_list.add_slice(slice);

            remaining_elem -= elems_in_mmap;

            if remaining_elem > 0 {
                current_file_num += 1;
                current_offset = 0;
                if current_file_num >= self.mmaps.len() {
                    return Err(anyhow!("Current file nunber out of bound"));
                }
            }
        }

        Ok(posting_list)
    }

    fn append(&mut self, posting_list: &[u64]) -> Result<()> {
        if self.current_num_of_posting_list == self.num_clusters {
            return Err(anyhow!(
                "Trying to append more posting lists than number of clusters"
            ));
        }
        let required_size = posting_list.len() * std::mem::size_of::<u64>();
        self.size_bytes += required_size;
        let should_flush = self.resident && self.size_bytes > self.memory_threshold;
        let flush = should_flush && !self.resident_posting_lists.is_empty();

        // Good case, where file is still resident
        if self.resident && !should_flush {
            self.current_num_of_posting_list += 1;
            self.resident_posting_lists.push(posting_list.to_vec());
            return Ok(());
        }

        // Spill to disk, creating new files if necessary
        let pl_metadata_in_bytes = PL_METADATA_LEN * std::mem::size_of::<u64>();
        if flush {
            self.flush_resident_posting_lists_to_disk()?;
            // At this point we are not in memory anymore, we'll need to
            // take into account the storage for metadata.
            self.size_bytes += self.resident_posting_lists.len() * pl_metadata_in_bytes;
        }

        self.size_bytes += pl_metadata_in_bytes;
        // We should spill to disk, but did not flush (there was nothing to flush)
        if self.resident && !flush {
            // Trigger the creation of a new file since this is the first time we write
            // to disk
            self.current_offset = self.backing_file_size;
            // Flip the flag since we are technically on disk at this point
            self.resident = false;
        } else {
            // Adjust the offset to write metadata first
            let metadata_offset = self.current_num_of_posting_list * pl_metadata_in_bytes;
            let file_access_info = self.offset_to_file_access_info(metadata_offset)?;
            self.current_offset = file_access_info.file_offset;
            self.current_backing_id = file_access_info.file_num as i32;
        }
        self.append_posting_list_to_disk(posting_list)?;
        Ok(())
    }

    fn write(&mut self, writer: &mut BufWriter<&mut File>) -> Result<usize> {
        let mut total_bytes_written = wrap_write(writer, &self.len().to_le_bytes())?;

        // If the data is still resident in memory, flush it to disk first
        if self.resident {
            self.flush_resident_posting_lists_to_disk()?;
        }

        for (i, mmap) in self.mmaps.iter().enumerate() {
            let bytes_to_write = if i as i32 == self.current_backing_id {
                self.current_offset
            } else {
                mmap.len()
            };

            let bytes_written = wrap_write(writer, &mmap[..bytes_to_write])?;
            total_bytes_written += bytes_written;

            if bytes_written != bytes_to_write {
                return Err(anyhow!(
                    "Failed to write entire mmap: expected {} bytes, wrote {} bytes",
                    bytes_to_write,
                    bytes_written
                ));
            }
        }

        writer.flush()?;

        Ok(total_bytes_written)
    }

    fn len(&self) -> usize {
        self.current_num_of_posting_list
    }

    fn config(&self) -> PostingListStorageConfig {
        PostingListStorageConfig {
            memory_threshold: self.memory_threshold,
            file_size: self.backing_file_size,
            num_clusters: self.num_clusters,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::io::{Read, Seek};

    use super::*;

    #[test]
    fn test_append_and_get_in_memory() {
        let tempdir = tempdir::TempDir::new("append_and_get_in_memory_test").unwrap();
        let base_directory = tempdir.path().to_str().unwrap().to_string();
        let mut storage = FileBackedAppendablePostingListStorage::new(
            base_directory,
            1024, // memory_threshold
            4096, // backing_file_size
            10,   // num_clusters
        );

        let pl1 = vec![1, 2, 3];
        let pl2 = vec![4, 5, 6, 7];
        assert!(storage.append(&pl1).is_ok());
        assert!(storage.append(&pl2).is_ok());
        assert!(storage.resident);
        assert_eq!(storage.resident_posting_lists.len(), 2);
        assert_eq!(
            storage.size_bytes,
            (pl1.len() + pl2.len()) * std::mem::size_of::<u64>()
        );
        assert_eq!(
            storage
                .get(0)
                .expect("Read back posting list should succeed")
                .iter()
                .collect::<Vec<_>>(),
            pl1
        );
        assert_eq!(
            storage
                .get(1)
                .expect("Read back posting list should succeed")
                .iter()
                .collect::<Vec<_>>(),
            pl2
        );
    }

    #[test]
    fn test_append_and_get_on_disk() {
        let tempdir = tempdir::TempDir::new("append_and_get_on_disk_test").unwrap();
        let base_directory = tempdir.path().to_str().unwrap().to_string();
        let mut storage = FileBackedAppendablePostingListStorage::new(
            base_directory,
            10,   // set a low threshold to force flushing to disk
            4096, // backing_file_size
            10,   // num_clusters
        );

        let pl1 = vec![1, 2, 3];
        let pl2 = vec![4, 5, 6, 7];
        let pl3 = vec![8, 9, 10];

        assert!(storage.append(&pl1).is_ok());
        assert!(storage.append(&pl2).is_ok());
        assert!(storage.append(&pl3).is_ok());

        assert!(!storage.resident);
        // Verify the content of the posting list
        assert_eq!(
            storage
                .get(0)
                .expect("Read back posting list should succeed")
                .iter()
                .collect::<Vec<_>>(),
            pl1
        );
        assert_eq!(
            storage
                .get(1)
                .expect("Read back posting list should succeed")
                .iter()
                .collect::<Vec<_>>(),
            pl2
        );
        assert_eq!(
            storage
                .get(2)
                .expect("Read back posting list should succeed")
                .iter()
                .collect::<Vec<_>>(),
            pl3
        );

        // Verify posting list metadata
        assert!(storage.mmaps.len() == 1);
        let size_in_bytes = std::mem::size_of::<u64>();
        let metadata_size = PL_METADATA_LEN * size_in_bytes; // length and offset
        let mmap = &storage.mmaps[0];

        // Read length
        let length_bytes: [u8; 8] = mmap[0..size_in_bytes].try_into().unwrap();
        let length = u64::from_le_bytes(length_bytes);
        assert_eq!(length, pl1.len() as u64);

        // Read offset
        let offset_bytes: [u8; 8] = mmap[size_in_bytes..metadata_size].try_into().unwrap();
        let offset = u64::from_le_bytes(offset_bytes);
        // Verify that the offset points to the correct location
        let expected_offset = storage.get_posting_list_offset();
        assert_eq!(offset, expected_offset as u64);
    }

    #[test]
    fn test_append_more_than_num_clusters() {
        let tempdir = tempdir::TempDir::new("append_more_than_num_clusters_test").unwrap();
        let base_directory = tempdir.path().to_str().unwrap().to_string();
        let mut storage = FileBackedAppendablePostingListStorage::new(
            base_directory,
            10,   // set a low threshold to force flushing to disk
            4096, // backing_file_size
            2,    // num_clusters
        );

        storage.append(&[1, 2, 3]).unwrap();
        storage.append(&[4, 5, 6]).unwrap();
        assert!(storage.append(&[7, 8, 9]).is_err());
    }

    #[test]
    fn test_get_out_of_bounds() {
        let tempdir = tempdir::TempDir::new("get_out_of_bounds_test").unwrap();
        let base_directory = tempdir.path().to_str().unwrap().to_string();
        let storage = FileBackedAppendablePostingListStorage::new(
            base_directory,
            10,   // set a low threshold to force flushing to disk
            4096, // backing_file_size
            2,    // num_clusters
        );
        assert!(storage.get(0).is_err());
    }

    #[test]
    fn test_append_and_get_across_multiple_mmaps() {
        let tempdir = tempdir::TempDir::new("append_and_get_across_multiple_mmaps_test").unwrap();
        let base_directory = tempdir.path().to_str().unwrap().to_string();
        let mut storage = FileBackedAppendablePostingListStorage::new(
            base_directory,
            10, // set a low threshold to force flushing to disk
            32, // small backing file size to force multiple mmaps
            2,  // num_clusters
        );

        let large_pl = (0..100).collect::<Vec<u64>>();
        storage.append(&large_pl).unwrap();

        // Verify the content of the posting list
        assert!(!storage.resident);
        let retrieved_pl = storage
            .get(0)
            .expect("Read back posting list should succeed")
            .iter()
            .collect::<Vec<_>>();
        assert_eq!(retrieved_pl, large_pl);

        // Verify that the posting list data spans multiple mmaps
        let size_in_bytes = std::mem::size_of::<u64>();
        let data_size = large_pl.len() * size_in_bytes;
        let first_mmap_data_size = storage.backing_file_size - storage.get_posting_list_offset();
        assert!(data_size > first_mmap_data_size);

        // Calculate how many mmaps should be used
        let expected_mmap_count = 1
            + (data_size - first_mmap_data_size + storage.backing_file_size - 1)
                / storage.backing_file_size;
        assert_eq!(storage.mmaps.len(), expected_mmap_count);
    }

    #[test]
    fn test_offset_to_file_access_info() {
        let tempdir = tempdir::TempDir::new("offset_to_file_access_info_test").unwrap();
        let base_directory = tempdir.path().to_str().unwrap().to_string();
        let mut storage =
            FileBackedAppendablePostingListStorage::new(base_directory, 1024, 4096, 2);
        storage
            .mmaps
            .push(memmap2::MmapMut::map_anon(4096).unwrap());
        storage
            .mmaps
            .push(memmap2::MmapMut::map_anon(4096).unwrap());

        let result = storage.offset_to_file_access_info(4500).unwrap();
        assert_eq!(result.file_num, 1);
        assert_eq!(result.file_offset, 404);

        assert!(storage.offset_to_file_access_info(10000).is_err());
    }

    #[test]
    fn test_flush_resident_posting_lists_to_disk() {
        let tempdir = tempdir::TempDir::new("flush_resident_posting_lists_to_disk_test").unwrap();
        let base_directory = tempdir.path().to_str().unwrap().to_string();
        let mut storage =
            FileBackedAppendablePostingListStorage::new(base_directory, 1024, 4096, 2);
        storage.append(&[1, 2, 3]).unwrap();
        storage.append(&[4, 5, 6]).unwrap();

        assert!(storage.resident);
        storage.flush_resident_posting_lists_to_disk().unwrap();
        assert!(!storage.resident);

        assert_eq!(
            storage
                .get(0)
                .expect("Read back posting list should succeed")
                .iter()
                .collect::<Vec<_>>(),
            &[1, 2, 3]
        );
        assert_eq!(
            storage
                .get(1)
                .expect("Read back posting list should succeed")
                .iter()
                .collect::<Vec<_>>(),
            &[4, 5, 6]
        );
    }

    #[test]
    fn test_append_posting_list_to_disk() {
        let tempdir = tempdir::TempDir::new("append_resident_posting_list_to_disk_test").unwrap();
        let base_directory = tempdir.path().to_str().unwrap().to_string();
        let mut storage =
            FileBackedAppendablePostingListStorage::new(base_directory, 1024, 4096, 2);
        // Trigger the creation of a new file
        storage.current_offset = storage.backing_file_size;
        storage.resident = false;
        storage.append_posting_list_to_disk(&[1, 2, 3]).unwrap();

        assert_eq!(
            storage
                .get(0)
                .expect("Read back posting list should succeed")
                .iter()
                .collect::<Vec<_>>(),
            &[1, 2, 3]
        );
    }

    #[test]
    fn test_append_and_get_large_posting_lists() {
        let tempdir = tempdir::TempDir::new("append_and_get_large_posting_lists_test").unwrap();
        let base_directory = tempdir.path().to_str().unwrap().to_string();

        let memory_threshold = 1024 * 1024; // 1 MB
        let backing_file_size = 512 * 1024; // 512 KB
        let num_clusters = 5;

        let mut storage = FileBackedAppendablePostingListStorage::new(
            base_directory,
            memory_threshold,
            backing_file_size,
            num_clusters,
        );

        // Create large posting lists that will span across multiple mmaps
        let large_posting_list_1: Vec<u64> = (0..100_000).collect();
        let large_posting_list_2: Vec<u64> = (100_000..200_000).collect();
        let large_posting_list_3: Vec<u64> = (200_000..300_000).collect();

        // Append large posting lists
        storage.append(&large_posting_list_1).unwrap();
        storage.append(&large_posting_list_2).unwrap();
        storage.append(&large_posting_list_3).unwrap();

        // Verify that the posting lists were appended correctly
        assert_eq!(storage.current_num_of_posting_list, 3);
        assert!(!storage.resident);

        // Retrieve and verify the posting lists
        let retrieved_list_1 = storage
            .get(0)
            .expect("Read back posting list should succeed")
            .iter()
            .collect::<Vec<_>>();
        let retrieved_list_2 = storage
            .get(1)
            .expect("Read back posting list should succeed")
            .iter()
            .collect::<Vec<_>>();
        let retrieved_list_3 = storage
            .get(2)
            .expect("Read back posting list should succeed")
            .iter()
            .collect::<Vec<_>>();

        assert_eq!(retrieved_list_1.len(), large_posting_list_1.len());
        assert_eq!(retrieved_list_2.len(), large_posting_list_2.len());
        assert_eq!(retrieved_list_3.len(), large_posting_list_3.len());

        for i in 0..large_posting_list_1.len() {
            assert_eq!(retrieved_list_1[i], large_posting_list_1[i]);
        }

        for i in 0..large_posting_list_2.len() {
            assert_eq!(retrieved_list_2[i], large_posting_list_2[i]);
        }

        for i in 0..large_posting_list_3.len() {
            assert_eq!(retrieved_list_3[i], large_posting_list_3[i]);
        }
    }

    #[test]
    fn test_write_and_verify() {
        let tempdir = tempdir::TempDir::new("write_and_verify_test").unwrap();
        let base_directory = tempdir.path().to_str().unwrap().to_string();
        let output_path = format!("{}/output", base_directory);
        let mut storage = FileBackedAppendablePostingListStorage::new(base_directory, 10, 4096, 2);

        let pl1 = vec![1, 2, 3];
        let pl2 = vec![4, 5, 6, 7];

        storage.append(&pl1).unwrap();
        storage.append(&pl2).unwrap();

        let mut output_file = fs::File::create(output_path.clone()).unwrap();
        let mut writer = std::io::BufWriter::new(&mut output_file);

        let bytes_written = storage.write(&mut writer).unwrap();
        assert!(bytes_written > 0);

        writer.flush().unwrap();

        // Reopen the file in read mode
        let mut output_file = OpenOptions::new().read(true).open(&output_path).unwrap();
        // Verify content
        output_file.seek(std::io::SeekFrom::Start(0)).unwrap();
        let mut content = Vec::new();
        output_file.read_to_end(&mut content).unwrap();
        assert_eq!(content.len(), bytes_written);

        output_file.seek(std::io::SeekFrom::Start(0)).unwrap();

        // Read and verify the data
        let mut reader = std::io::BufReader::new(&output_file);

        // Read number of clusters
        let mut num_clusters_bytes = [0u8; std::mem::size_of::<u64>()];
        reader.read_exact(&mut num_clusters_bytes).unwrap();
        let num_clusters = u64::from_le_bytes(num_clusters_bytes);
        assert_eq!(num_clusters, storage.num_clusters as u64);

        // Read metadata for each posting list
        for i in 0..PL_METADATA_LEN {
            let mut length_bytes = [0u8; std::mem::size_of::<u64>()];
            let mut offset_bytes = [0u8; std::mem::size_of::<u64>()];
            reader.read_exact(&mut length_bytes).unwrap();
            reader.read_exact(&mut offset_bytes).unwrap();

            let length = u64::from_le_bytes(length_bytes);
            let offset = u64::from_le_bytes(offset_bytes);

            assert_eq!(length as usize, if i == 0 { pl1.len() } else { pl2.len() });
            assert!(offset > 0);
        }

        // Read and verify posting lists
        let mut buffer = Vec::new();
        reader.read_to_end(&mut buffer).unwrap();

        let pl1_bytes: Vec<u8> = pl1.iter().flat_map(|&x| x.to_le_bytes().to_vec()).collect();
        let pl2_bytes: Vec<u8> = pl2.iter().flat_map(|&x| x.to_le_bytes().to_vec()).collect();

        assert!(buffer
            .windows(pl1_bytes.len())
            .any(|window| window == pl1_bytes.as_slice()));
        assert!(buffer
            .windows(pl2_bytes.len())
            .any(|window| window == pl2_bytes.as_slice()));
    }
}
