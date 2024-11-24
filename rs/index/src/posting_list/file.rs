use std::fs::OpenOptions;
use std::io::Write;
use std::vec;

use anyhow::{anyhow, Result};
use utils::mem::{transmute_slice_to_u8, transmute_u8_to_val};

use super::{PostingList, PostingListStorage, PostingListStorageConfig};

#[derive(Debug)]
struct FileAccessId {
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
    resident_posting_lists: Vec<Vec<usize>>,
    resident: bool,

    // Only has value if we spill to disk
    base_directory: String,
    mmaps: Vec<memmap2::MmapMut>,
    current_backing_id: i32,
    current_offset: usize,
    offset_to_current_posting_list: usize,
}

impl FileBackedAppendablePostingListStorage {
    pub fn new(
        base_directory: String,
        memory_threshold: usize,
        backing_file_size: usize,
        num_clusters: usize,
    ) -> Self {
        let pl_metadata_in_bytes = 2 * std::mem::size_of::<usize>();
        let offset_to_current_posting_list = num_clusters * pl_metadata_in_bytes;
        // Rounding to 2 * `usize` size in bytes to at least simplify the reading of
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

    fn offset_to_first_posting_list(&self) -> usize {
        let pl_metadata_in_bytes = 2 * std::mem::size_of::<usize>();
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
        let usize_in_bytes = std::mem::size_of::<usize>();
        // Extract all the data we need from self to avoid immutable borrow issue
        let posting_lists: Vec<Vec<usize>> = std::mem::take(&mut self.resident_posting_lists);
        // First write the metadata
        let mut buffer: Vec<u8> = vec![];
        for posting_list in &posting_lists {
            if self.current_offset == self.backing_file_size {
                self.new_backing_file()?;
            }
            buffer.extend_from_slice(posting_list.len().to_le_bytes().as_ref());
            buffer.extend_from_slice(self.offset_to_current_posting_list.to_le_bytes().as_ref());
            self.write_to_current_mmap(&buffer)?;
            self.offset_to_current_posting_list += posting_list.len() * usize_in_bytes;
            buffer.clear();
        }
        // Now write the posting lists
        self.current_offset = self.offset_to_first_posting_list();
        for posting_list in &posting_lists {
            for idx in posting_list {
                if self.current_offset == self.backing_file_size {
                    self.new_backing_file()?;
                }
                buffer.extend_from_slice(idx.to_le_bytes().as_ref());
                // Write as much as possible to the current backing file
                let available_space = self.backing_file_size - self.current_offset;
                if buffer.len() == available_space {
                    self.write_to_current_mmap(&buffer)?;
                    buffer.clear();
                }
            }
        }
        // Write any remaining data in the buffer after processing all posting lists
        if !buffer.is_empty() {
            self.write_to_current_mmap(&buffer)?;
            buffer.clear();
        }
        self.resident = false;
        self.resident_posting_lists.clear();
        Ok(())
    }

    // The caller is responsible for setting self.current_offset to the right value
    fn append_posting_list_to_disk(&mut self, posting_list: &[usize]) -> Result<()> {
        if self.resident {
            return Err(anyhow!("Posting lists should already be flushed to disk"));
        }
        let usize_in_bytes = std::mem::size_of::<usize>();
        // First write the metadata
        if self.current_offset == self.backing_file_size {
            self.new_backing_file()?;
        }
        let mut buffer: Vec<u8> = vec![];
        buffer.extend_from_slice(posting_list.len().to_le_bytes().as_ref());
        buffer.extend_from_slice(self.offset_to_current_posting_list.to_le_bytes().as_ref());
        self.write_to_current_mmap(&buffer)?;
        buffer.clear();

        // Now write the posting list
        self.current_offset = self.offset_to_current_posting_list;
        self.offset_to_current_posting_list += posting_list.len() * usize_in_bytes;
        for idx in posting_list.iter() {
            if self.current_offset == self.backing_file_size {
                self.new_backing_file()?;
            }
            buffer.extend_from_slice(idx.to_le_bytes().as_ref());
            // Write as much as possible to the current backing file
            let available_space = self.backing_file_size - self.current_offset;
            if buffer.len() == available_space {
                self.write_to_current_mmap(&buffer)?;
                buffer.clear();
            }
        }
        // Write any remaining data in the buffer after processing all posting lists
        if !buffer.is_empty() {
            self.write_to_current_mmap(&buffer)?;
            buffer.clear();
        }
        self.current_num_of_posting_list += 1;
        Ok(())
    }

    fn offset_to_file_num_and_file_offset(&self, offset: usize) -> Result<FileAccessId> {
        let file_num = offset / self.backing_file_size;
        if file_num >= self.mmaps.len() {
            return Err(anyhow!("File number out of bound"));
        }

        let file_offset = offset % self.backing_file_size;
        if file_offset >= self.mmaps[file_num].len() {
            return Err(anyhow!("File offset out of bound"));
        }

        Ok(FileAccessId {
            file_num,
            file_offset,
        })
    }
}

impl<'a> PostingListStorage<'a> for FileBackedAppendablePostingListStorage {
    fn get(&'a self, id: u32) -> Result<PostingList<'a>> {
        let i = id as usize;
        let usize_in_bytes = std::mem::size_of::<usize>();

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
        let offset_to_pl_metadata = i * 2 * usize_in_bytes;

        let file_access_id = self.offset_to_file_num_and_file_offset(offset_to_pl_metadata)?;
        let mmap = &self.mmaps[file_access_id.file_num];
        let slice = &mmap[file_access_id.file_offset..file_access_id.file_offset + usize_in_bytes];
        let pl_len = transmute_u8_to_val(slice);
        let slice = &mmap[file_access_id.file_offset + usize_in_bytes
            ..file_access_id.file_offset + 2 * usize_in_bytes];
        let pl_offset = transmute_u8_to_val(slice);

        let file_access_id = self.offset_to_file_num_and_file_offset(pl_offset)?;
        let required_size = pl_len * usize_in_bytes;

        // Posting list fits within a single mmap
        if file_access_id.file_offset + required_size <= mmap.len() {
            let mmap = &self.mmaps[file_access_id.file_num];
            let slice =
                &mmap[file_access_id.file_offset..file_access_id.file_offset + required_size];
            return Ok(PostingList::new_with_slices(vec![slice]));
        }

        // Posting list spans across multiple mmaps.
        let mut posting_list = PostingList::new();
        let mut remaining_elem = pl_len;
        let mut current_file_num = file_access_id.file_num;
        let mut current_offset = file_access_id.file_offset;
        while remaining_elem > 0 {
            let mmap = &self.mmaps[current_file_num];
            let bytes_left_in_mmap = mmap.len() - current_offset;
            let elems_in_mmap = std::cmp::min(remaining_elem, bytes_left_in_mmap / usize_in_bytes);

            let slice = &mmap[current_offset..current_offset + elems_in_mmap * usize_in_bytes];
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

    fn append(&mut self, posting_list: &[usize]) -> Result<()> {
        if self.current_num_of_posting_list == self.num_clusters {
            return Err(anyhow!(
                "Trying to append more posting lists than number of clusters"
            ));
        }
        let required_size = posting_list.len() * std::mem::size_of::<usize>();
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
        let pl_metadata_in_bytes = 2 * std::mem::size_of::<usize>();
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
            self.current_offset = self.current_num_of_posting_list * pl_metadata_in_bytes;
        }
        self.append_posting_list_to_disk(posting_list)?;
        Ok(())
    }

    fn write(&mut self, writer: &mut std::io::BufWriter<&mut std::fs::File>) -> Result<usize> {
        let mut total_bytes_written = writer.write(&self.len().to_le_bytes())?;

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

            let bytes_written = writer.write(&mmap[..bytes_to_write])?;
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
            (pl1.len() + pl2.len()) * std::mem::size_of::<usize>()
        );
        assert_eq!(
            storage
                .get(0)
                .expect("Read back posting list should succeed")
                .collect::<Vec<_>>(),
            pl1
        );
        assert_eq!(
            storage
                .get(1)
                .expect("Read back posting list should succeed")
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
                .collect::<Vec<_>>(),
            pl1
        );
        assert_eq!(
            storage
                .get(1)
                .expect("Read back posting list should succeed")
                .collect::<Vec<_>>(),
            pl2
        );
        assert_eq!(
            storage
                .get(2)
                .expect("Read back posting list should succeed")
                .collect::<Vec<_>>(),
            pl3
        );

        // Verify posting list metadata
        assert!(storage.mmaps.len() == 1);
        let usize_size = std::mem::size_of::<usize>();
        let metadata_size = 2 * usize_size; // length and offset
        let mmap = &storage.mmaps[0];

        // Read length
        let length_bytes: [u8; 8] = mmap[0..usize_size].try_into().unwrap();
        let length = usize::from_le_bytes(length_bytes);
        assert_eq!(length, pl1.len());

        // Read offset
        let offset_bytes: [u8; 8] = mmap[usize_size..metadata_size].try_into().unwrap();
        let offset = usize::from_le_bytes(offset_bytes);
        // Verify that the offset points to the correct location
        let expected_offset = storage.offset_to_first_posting_list();
        assert_eq!(offset, expected_offset);
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

        let large_pl = (0..100).collect::<Vec<usize>>();
        storage.append(&large_pl).unwrap();

        // Verify the content of the posting list
        assert!(!storage.resident);
        let retrieved_pl = storage
            .get(0)
            .expect("Read back posting list should succeed")
            .collect::<Vec<_>>();
        assert_eq!(retrieved_pl, large_pl);

        // Verify that the posting list data spans multiple mmaps
        let usize_in_bytes = std::mem::size_of::<usize>();
        let data_size = large_pl.len() * usize_in_bytes;
        let first_mmap_data_size =
            storage.backing_file_size - storage.offset_to_first_posting_list();
        assert!(data_size > first_mmap_data_size);

        // Calculate how many mmaps should be used
        let expected_mmap_count = 1
            + (data_size - first_mmap_data_size + storage.backing_file_size - 1)
                / storage.backing_file_size;
        assert_eq!(storage.mmaps.len(), expected_mmap_count);
    }

    #[test]
    fn test_offset_to_file_num_and_file_offset() {
        let tempdir = tempdir::TempDir::new("offset_to_file_num_and_file_offset_test").unwrap();
        let base_directory = tempdir.path().to_str().unwrap().to_string();
        let mut storage =
            FileBackedAppendablePostingListStorage::new(base_directory, 1024, 4096, 2);
        storage
            .mmaps
            .push(memmap2::MmapMut::map_anon(4096).unwrap());
        storage
            .mmaps
            .push(memmap2::MmapMut::map_anon(4096).unwrap());

        let result = storage.offset_to_file_num_and_file_offset(4500).unwrap();
        assert_eq!(result.file_num, 1);
        assert_eq!(result.file_offset, 404);

        assert!(storage.offset_to_file_num_and_file_offset(10000).is_err());
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
                .collect::<Vec<_>>(),
            &[1, 2, 3]
        );
        assert_eq!(
            storage
                .get(1)
                .expect("Read back posting list should succeed")
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
                .collect::<Vec<_>>(),
            &[1, 2, 3]
        );
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
        let mut num_clusters_bytes = [0u8; std::mem::size_of::<usize>()];
        reader.read_exact(&mut num_clusters_bytes).unwrap();
        let num_clusters = usize::from_le_bytes(num_clusters_bytes);
        assert_eq!(num_clusters, storage.num_clusters);

        // Read metadata for each posting list
        for i in 0..2 {
            let mut length_bytes = [0u8; std::mem::size_of::<usize>()];
            let mut offset_bytes = [0u8; std::mem::size_of::<usize>()];
            reader.read_exact(&mut length_bytes).unwrap();
            reader.read_exact(&mut offset_bytes).unwrap();

            let length = usize::from_le_bytes(length_bytes);
            let offset = usize::from_le_bytes(offset_bytes);

            assert_eq!(length, if i == 0 { pl1.len() } else { pl2.len() });
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
