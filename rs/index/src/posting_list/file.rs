use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::mem::size_of;
use std::vec;

use anyhow::{anyhow, Result};
use utils::io::wrap_write;
use utils::mem::transmute_slice_to_u8;

use super::{PostingList, PostingListStorageConfig};

const PL_METADATA_LEN: usize = 2;

#[derive(Debug)]
struct FileAccessInfo {
    file_num: usize,
    file_offset: usize,
}

struct BackingFiles {
    mmaps: Vec<memmap2::MmapMut>,
    base_directory: String,
    file_suffix: String,
    file_size: usize,
    current_overall_offset: usize,
}

impl BackingFiles {
    pub fn flush(&mut self) -> Result<()> {
        // Flush all mmaps
        for mmap in self.mmaps.iter_mut() {
            mmap.flush()?;
        }
        Ok(())
    }

    pub fn is_current_backing_file_full(&self) -> bool {
        self.current_overall_offset.is_multiple_of(self.file_size)
    }

    pub fn get_current_offset(&self) -> usize {
        self.current_overall_offset
    }

    pub fn new_backing_file(&mut self) -> Result<()> {
        let backing_file_size = self.file_size as u64;

        let file_name = format!(
            "{}/{}.bin.{}",
            &self.base_directory,
            self.file_suffix,
            self.mmaps.len()
        );

        let backing_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(file_name)?;

        backing_file.set_len(backing_file_size)?;

        self.mmaps
            .push(unsafe { memmap2::MmapMut::map_mut(&backing_file)? });

        Ok(())
    }

    pub fn offset_to_file_access_info(&self, offset: usize) -> Result<FileAccessInfo> {
        let file_num = offset / self.file_size;
        if file_num >= self.mmaps.len() {
            return Err(anyhow!("File number out of bound"));
        }

        let file_offset = offset % self.file_size;
        if file_offset >= self.mmaps[file_num].len() {
            return Err(anyhow!("File offset out of bound"));
        }

        Ok(FileAccessInfo {
            file_num,
            file_offset,
        })
    }

    // The caller has to ensure data fits in the current mmap (so it also takes care of
    // creating new file if needed).
    // TODO(tyb): make this function handle these cases instead of the caller.
    pub fn write_to_current_mmap(&mut self, data: &[u8]) -> Result<()> {
        let file_offset = self.current_overall_offset % self.file_size;
        let write_size = data.len();
        let current_mmap = &mut self.mmaps.last_mut();

        match current_mmap {
            Some(mmap) => {
                mmap[file_offset..file_offset + write_size].copy_from_slice(data);
                self.current_overall_offset += write_size;
                Ok(())
            }
            None => Err(anyhow!("Cannot get current mmap")),
        }
    }

    pub fn get_slices_at(&self, start_offset: usize, size_in_bytes: usize) -> Result<Vec<&[u8]>> {
        let file_access_info = self.offset_to_file_access_info(start_offset)?;
        let (file_offset, file_num) = (file_access_info.file_offset, file_access_info.file_num);

        let mmap = &self.mmaps[file_num];

        // Requested slice fits within a single mmap
        if file_offset + size_in_bytes <= self.file_size {
            let slice = &mmap[file_offset..file_offset + size_in_bytes];
            return Ok(vec![slice]);
        }

        // Requested slice spans across multiple mmaps
        let mut slices = Vec::new();
        let mut remaining_bytes = size_in_bytes;
        let mut current_file_num = file_num;
        let mut current_offset = file_offset;
        while remaining_bytes > 0 {
            let mmap = &self.mmaps[current_file_num];
            let bytes_left_in_mmap = self.file_size - current_offset;
            let bytes_to_read = std::cmp::min(remaining_bytes, bytes_left_in_mmap);

            let slice = &mmap[current_offset..current_offset + bytes_to_read];
            slices.push(slice);

            remaining_bytes -= bytes_to_read;

            if remaining_bytes > 0 {
                current_file_num += 1;
                current_offset = 0;
                if current_file_num >= self.mmaps.len() {
                    return Err(anyhow!("Current file number out of bound"));
                }
            }
        }
        Ok(slices)
    }

    fn write_mmaps_to_file(&self, writer: &mut BufWriter<&mut File>) -> Result<usize> {
        let file_access_info = self.offset_to_file_access_info(self.current_overall_offset)?;
        let (file_offset, file_num) = (file_access_info.file_offset, file_access_info.file_num);

        let mut total_bytes_written = 0;
        for (i, mmap) in self.mmaps.iter().enumerate() {
            let bytes_to_write = if i == file_num {
                file_offset
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

        Ok(total_bytes_written)
    }
}

pub struct FileBackedAppendablePostingListStorage {
    pub memory_threshold: usize,
    pub backing_file_size: usize,
    // Counter of appended posting lists.
    entry_count: usize,

    // Number of bytes required to store the posting lists in memory.
    size_bytes: usize,
    resident_posting_lists: Vec<Vec<u64>>,

    // Whether it's currently in memory
    resident: bool,

    // Backing files for posting list metadata
    metadata_backing_files: BackingFiles,
    // Backing files for posting lists themselves
    posting_list_backing_files: BackingFiles,
    // We currently use overall offset since it would be directly usable by queries to find
    // the appropriate posting list. We could also use (backing_id, offset_in_file) pair to
    // track current positions, but we'll need to convert that back to overall offset on
    // storage's `write`.
    // TODO(tyb): maybe consider using (backing_id, offset_in_file) pair for consistency with
    // FileBackedAppendableVectorStorage. Or switch to overall offset for both.
}

impl FileBackedAppendablePostingListStorage {
    pub fn new(base_directory: String, memory_threshold: usize, backing_file_size: usize) -> Self {
        let pl_metadata_in_bytes = PL_METADATA_LEN * size_of::<u64>();
        // Rounding to PL_METADATA_LEN * `u64` size in bytes to at least simplify the reading of
        // posting list offsets and lengths.
        //
        // That's the best we can do since we do not know sizes of posting lists
        // beforehand.
        let rounded_backing_file_size =
            backing_file_size / pl_metadata_in_bytes * pl_metadata_in_bytes;
        Self {
            memory_threshold,
            backing_file_size: rounded_backing_file_size,
            entry_count: 0,
            size_bytes: 0,
            resident_posting_lists: vec![],
            resident: true,
            metadata_backing_files: BackingFiles {
                mmaps: vec![],
                base_directory: base_directory.clone(),
                file_suffix: "metadata".to_string(),
                file_size: rounded_backing_file_size,
                current_overall_offset: 0,
            },
            posting_list_backing_files: BackingFiles {
                mmaps: vec![],
                base_directory: base_directory.clone(),
                file_suffix: "posting_list".to_string(),
                file_size: rounded_backing_file_size,
                current_overall_offset: 0,
            },
        }
    }

    pub fn new_with_config(base_directory: String, config: PostingListStorageConfig) -> Self {
        Self::new(base_directory, config.memory_threshold, config.file_size)
    }

    pub fn is_resident(&self) -> bool {
        self.resident
    }

    pub fn flush(&mut self) -> Result<()> {
        if !self.resident {
            self.metadata_backing_files.flush()?;
            self.posting_list_backing_files.flush()?;
        }
        Ok(())
    }

    fn write_posting_list_to_disk(&mut self, posting_list: &[u64]) -> Result<()> {
        // First write the metadata
        if self.metadata_backing_files.is_current_backing_file_full() {
            self.metadata_backing_files.new_backing_file()?;
        }
        // Write the length of the posting list
        self.metadata_backing_files
            .write_to_current_mmap(&((size_of_val(posting_list)) as u64).to_le_bytes())?;
        // Write the offset to the current posting list
        self.metadata_backing_files.write_to_current_mmap(
            &self
                .posting_list_backing_files
                .get_current_offset()
                .to_le_bytes(),
        )?;

        // Now write the posting list itself
        for idx in posting_list {
            if self
                .posting_list_backing_files
                .is_current_backing_file_full()
            {
                self.posting_list_backing_files.new_backing_file()?;
            }
            self.posting_list_backing_files
                .write_to_current_mmap(&idx.to_le_bytes())?;
        }
        Ok(())
    }

    fn flush_resident_posting_lists_to_disk(&mut self) -> Result<()> {
        if !self.resident {
            return Err(anyhow!(
                "Posting lists should still be in memory when flushing to disk"
            ));
        }

        // Extract all the data we need from self to avoid immutable borrow issue
        let posting_lists: Vec<Vec<u64>> = std::mem::take(&mut self.resident_posting_lists);

        for posting_list in &posting_lists {
            self.write_posting_list_to_disk(posting_list)?;
        }

        self.resident = false;
        self.resident_posting_lists.clear();
        Ok(())
    }

    // The caller is responsible for setting self.resident to false.
    fn append_posting_list_to_disk(&mut self, posting_list: &[u64]) -> Result<()> {
        if self.resident {
            return Err(anyhow!("Posting lists should already be flushed to disk"));
        }

        self.write_posting_list_to_disk(posting_list)?;

        self.entry_count += 1;
        Ok(())
    }
}

impl<'a> FileBackedAppendablePostingListStorage {
    pub fn get(&'a self, id: u32) -> Result<PostingList<'a>> {
        let i = id as usize;

        if self.resident {
            if i >= self.len() {
                return Err(anyhow!(
                    "Posting list id {} out of bound (current len {})",
                    i,
                    self.len()
                ));
            }
            return PostingList::new_with_slices(vec![transmute_slice_to_u8(
                &self.resident_posting_lists[i],
            )]);
        }

        if i >= self.entry_count {
            return Err(anyhow!("Posting list id out of bound"));
        }
        let u64_bytes = size_of::<u64>();
        let metadata_len = PL_METADATA_LEN * u64_bytes;
        let metadata_offset = i * metadata_len;

        let metadata_slice = self
            .metadata_backing_files
            .get_slices_at(metadata_offset, metadata_len)?
            .into_iter()
            .next()
            .ok_or(anyhow!("Expected a single slice but got none"))?;

        let required_size = u64::from_le_bytes(metadata_slice[..u64_bytes].try_into()?) as usize;
        let pl_offset = u64::from_le_bytes(metadata_slice[u64_bytes..].try_into()?) as usize;

        PostingList::new_with_slices(
            self.posting_list_backing_files
                .get_slices_at(pl_offset, required_size)?,
        )
    }

    pub fn append(&mut self, posting_list: &[u64]) -> Result<()> {
        let required_size = size_of_val(posting_list);
        self.size_bytes += required_size;
        let should_flush = self.resident && self.size_bytes > self.memory_threshold;
        let flush = should_flush && !self.resident_posting_lists.is_empty();

        // Good case, where file is still resident
        if self.resident && !should_flush {
            self.entry_count += 1;
            self.resident_posting_lists.push(posting_list.to_vec());
            return Ok(());
        }

        // Spill to disk, creating new files if necessary
        if flush {
            self.flush_resident_posting_lists_to_disk()?;
        }

        // We should spill to disk, but did not flush (there was nothing to flush)
        if self.resident && !flush {
            // Flip the flag since we are technically on disk at this point
            self.resident = false;
        }
        self.append_posting_list_to_disk(posting_list)?;
        Ok(())
    }

    pub fn write(&mut self, writer: &mut BufWriter<&mut File>) -> Result<usize> {
        let mut total_bytes_written = wrap_write(writer, &self.len().to_le_bytes())?;

        // If the data is still resident in memory, flush it to disk first
        if self.resident {
            self.flush_resident_posting_lists_to_disk()?;
        }

        // First write the metadata
        total_bytes_written += self.metadata_backing_files.write_mmaps_to_file(writer)?;

        // Now write the posting lists
        total_bytes_written += self
            .posting_list_backing_files
            .write_mmaps_to_file(writer)?;

        writer.flush()?;

        Ok(total_bytes_written)
    }

    pub fn len(&self) -> usize {
        self.entry_count
    }

    pub fn is_empty(&self) -> bool {
        self.entry_count == 0
    }

    pub fn config(&self) -> PostingListStorageConfig {
        PostingListStorageConfig {
            memory_threshold: self.memory_threshold,
            file_size: self.backing_file_size,
            num_clusters: self.entry_count,
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
        );

        let pl1 = vec![1, 2, 3];
        let pl2 = vec![4, 5, 6, 7];
        assert!(storage.append(&pl1).is_ok());
        assert!(storage.append(&pl2).is_ok());
        assert!(storage.resident);
        assert_eq!(storage.resident_posting_lists.len(), 2);
        assert_eq!(
            storage.size_bytes,
            (pl1.len() + pl2.len()) * size_of::<u64>()
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
        assert!(storage.metadata_backing_files.mmaps.len() == 1);
        let u64_bytes = size_of::<u64>();
        let metadata_size = PL_METADATA_LEN * u64_bytes; // length and offset
        let mmap = &storage.metadata_backing_files.mmaps[0];

        // Read length
        let length_bytes: [u8; 8] = mmap[0..u64_bytes].try_into().unwrap();
        let length = u64::from_le_bytes(length_bytes);
        assert_eq!(length, (pl1.len() * u64_bytes) as u64);

        // Read offset
        let offset_bytes: [u8; 8] = mmap[u64_bytes..metadata_size].try_into().unwrap();
        let offset = u64::from_le_bytes(offset_bytes);
        // Verify that the offset points to the correct location
        assert_eq!(offset, 0);
    }

    #[test]
    fn test_get_out_of_bounds() {
        let tempdir = tempdir::TempDir::new("get_out_of_bounds_test").unwrap();
        let base_directory = tempdir.path().to_str().unwrap().to_string();
        let storage = FileBackedAppendablePostingListStorage::new(
            base_directory,
            10,   // set a low threshold to force flushing to disk
            4096, // backing_file_size
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
        let u64_bytes = size_of::<u64>();
        let data_size = large_pl.len() * u64_bytes;
        assert!(data_size > storage.backing_file_size);

        // Calculate how many mmaps should be used
        let expected_mmap_count = data_size.div_ceil(storage.backing_file_size);
        assert_eq!(
            storage.posting_list_backing_files.mmaps.len(),
            expected_mmap_count
        );
    }

    #[test]
    fn test_offset_to_file_access_info() {
        let tempdir = tempdir::TempDir::new("offset_to_file_access_info_test").unwrap();
        let base_directory = tempdir.path().to_str().unwrap().to_string();
        let mut backing_files = BackingFiles {
            mmaps: vec![],
            base_directory,
            file_suffix: "test".to_string(),
            file_size: 4096,
            current_overall_offset: 0,
        };
        backing_files
            .mmaps
            .push(memmap2::MmapMut::map_anon(4096).unwrap());
        backing_files
            .mmaps
            .push(memmap2::MmapMut::map_anon(4096).unwrap());

        let result = backing_files.offset_to_file_access_info(4500).unwrap();
        assert_eq!(result.file_num, 1);
        assert_eq!(result.file_offset, 404);

        assert!(backing_files.offset_to_file_access_info(10000,).is_err());
    }

    #[test]
    fn test_flush_resident_posting_lists_to_disk() {
        let tempdir = tempdir::TempDir::new("flush_resident_posting_lists_to_disk_test").unwrap();
        let base_directory = tempdir.path().to_str().unwrap().to_string();
        let mut storage = FileBackedAppendablePostingListStorage::new(base_directory, 1024, 4096);
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
        let mut storage = FileBackedAppendablePostingListStorage::new(base_directory, 1024, 4096);
        // Trigger the creation of a new file
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

        let mut storage = FileBackedAppendablePostingListStorage::new(
            base_directory,
            memory_threshold,
            backing_file_size,
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
        assert_eq!(storage.entry_count, 3);
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
        let mut storage = FileBackedAppendablePostingListStorage::new(base_directory, 10, 4096);

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
        let mut num_clusters_bytes = [0u8; size_of::<u64>()];
        reader.read_exact(&mut num_clusters_bytes).unwrap();
        let num_clusters = u64::from_le_bytes(num_clusters_bytes);
        assert_eq!(num_clusters, storage.len() as u64);

        // Read metadata for each posting list
        for i in 0..storage.len() {
            let mut length_bytes = [0u8; size_of::<u64>()];
            let mut offset_bytes = [0u8; size_of::<u64>()];
            reader.read_exact(&mut length_bytes).unwrap();
            reader.read_exact(&mut offset_bytes).unwrap();

            let length = u64::from_le_bytes(length_bytes);
            let offset = u64::from_le_bytes(offset_bytes);

            assert_eq!(
                length as usize,
                if i == 0 {
                    pl1.len() * size_of::<u64>()
                } else {
                    pl2.len() * size_of::<u64>()
                }
            );
            assert_eq!(
                offset as usize,
                if i == 0 {
                    0
                } else {
                    pl1.len() * size_of::<u64>()
                }
            );
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
