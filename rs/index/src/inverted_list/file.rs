use std::borrow::Cow;
use std::fs::OpenOptions;
use std::io::Write;
use std::vec;

use anyhow::{anyhow, Result};
use num_traits::ToBytes;
use utils::io::wrap_write;

use super::InvertedListStorageConfig;

pub struct Centroid<T> {
    pub vector: Vec<T>,
    pub posting_list_len: usize,
}

pub struct FileBackedAppendableInvertedListStorage<T> {
    pub memory_threshold: usize,
    pub backing_file_size: usize,
    num_features: usize,
    num_clusters: usize,

    // Whether it's currently in memory
    resident_centroids: Vec<Centroid<T>>,
    resident_posting_lists: Vec<Vec<usize>>,
    resident: bool,
    size_bytes: usize,

    // Only has value if we spill to disk
    base_directory: String,
    mmaps: Vec<memmap2::MmapMut>,
    current_backing_id: i32,
    current_offset: usize,
    current_posting_list_offset: usize,
}

impl<T: ToBytes + Clone> FileBackedAppendableInvertedListStorage<T> {
    pub fn new(
        base_directory: String,
        memory_threshold: usize,
        backing_file_size: usize,
        num_features: usize,
        num_clusters: usize,
    ) -> Self {
        let current_posting_list_offset =
            num_clusters * (num_features * std::mem::size_of::<T>() + std::mem::size_of::<usize>());
        let bytes_per_centroid =
            num_features * std::mem::size_of::<T>() + 2 * std::mem::size_of::<usize>();
        // Only rounding to centroid size to simplify flushing resident centroids to disk.
        // We can't do the same for posting lists because  we don't know their sizes beforehand.
        //
        // However, we know that rounded_backing_file_size is a multiple of
        // std::mem::size_of::<usize>(), so that we can always write an entire vector index from
        // the posting list to the mmap.
        let rounded_backing_file_size = backing_file_size / bytes_per_centroid * bytes_per_centroid;
        Self {
            base_directory,
            memory_threshold,
            backing_file_size: rounded_backing_file_size,
            num_features,
            num_clusters,
            resident_centroids: vec![],
            resident_posting_lists: vec![],
            resident: true,
            size_bytes: 0,
            mmaps: vec![],
            current_backing_id: -1,
            current_offset: 0,
            current_posting_list_offset,
        }
    }

    pub fn new_with_config(base_directory: String, config: InvertedListStorageConfig) -> Self {
        Self::new(
            base_directory,
            config.memory_threshold,
            config.file_size,
            config.num_features,
            config.num_clusters,
        )
    }

    pub fn is_resident(&self) -> bool {
        self.resident
    }

    pub fn flush(&mut self) -> Result<(), String> {
        if self.resident {
            return Ok(());
        }

        // Flush all mmaps
        for mmap in self.mmaps.iter_mut() {
            let res = mmap.flush();
            if res.is_err() {
                return Err(res.err().unwrap().to_string());
            }
        }
        Ok(())
    }

    fn new_backing_file(&mut self) {
        self.current_backing_id += 1;
        let backing_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(format!(
                "{}/inverted_list.bin.{}",
                self.base_directory, self.current_backing_id
            ))
            .unwrap();

        backing_file.set_len(self.backing_file_size as u64).unwrap();

        self.mmaps
            .push(unsafe { memmap2::MmapMut::map_mut(&backing_file).unwrap() });
        self.current_offset = 0;
    }

    fn flush_resident_centroids_to_disk(&mut self) {
        assert!(self.resident);
        self.current_offset = self.backing_file_size;
        let len = self.resident_centroids.len();
        for i in 0..len {
            if self.current_offset == self.backing_file_size {
                self.new_backing_file();
            }
            self.append_resident_centroid_to_current_mmap(i);
        }
        self.resident = false;
        self.resident_centroids.clear();
    }

    // Caller is responsible for preparing new mmaps if necessary.
    fn append_resident_centroid_to_current_mmap(&mut self, resident_idx: usize) {
        let centroid = &self.resident_centroids[resident_idx];
        let size_required =
            self.num_features * std::mem::size_of::<T>() + 2 * std::mem::size_of::<usize>();
        let mut buffer: Vec<u8> = vec![];
        for i in 0..centroid.vector.len() {
            buffer.extend_from_slice(centroid.vector[i].to_le_bytes().as_ref());
        }
        buffer.extend_from_slice(centroid.posting_list_len.to_le_bytes().as_ref());
        buffer.extend_from_slice(self.current_posting_list_offset.to_le_bytes().as_ref());
        let size_required_for_posting_list =
            centroid.posting_list_len * std::mem::size_of::<usize>();
        self.current_posting_list_offset += size_required_for_posting_list;

        let mmap = &mut self.mmaps[self.current_backing_id as usize];
        // Copy buffer to current offset in mmap
        mmap[self.current_offset..self.current_offset + size_required].copy_from_slice(&buffer);
        self.current_offset += size_required;
    }

    // TODO(tyb): Factor out the writing to mmap, revisiting the naming too.
    fn append_centroid_to_disk(&mut self, centroid: Centroid<T>) {
        if self.current_offset == self.backing_file_size {
            self.new_backing_file();
        }
        let size_required =
            self.num_features * std::mem::size_of::<T>() + 2 * std::mem::size_of::<usize>();
        assert!(self.current_offset + size_required <= self.backing_file_size);
        let mut buffer: Vec<u8> = vec![];
        for i in 0..centroid.vector.len() {
            buffer.extend_from_slice(centroid.vector[i].to_le_bytes().as_ref());
        }
        buffer.extend_from_slice(centroid.posting_list_len.to_le_bytes().as_ref());
        buffer.extend_from_slice(self.current_posting_list_offset.to_le_bytes().as_ref());
        let size_required_for_posting_list =
            centroid.posting_list_len * std::mem::size_of::<usize>();
        self.current_posting_list_offset += size_required_for_posting_list;

        let mmap = &mut self.mmaps[self.current_backing_id as usize];
        // Copy buffer to current offset in mmap
        mmap[self.current_offset..self.current_offset + size_required].copy_from_slice(&buffer);
        self.current_offset += size_required;
    }

    pub fn get_centroid(&self, id: u32) -> Result<(&[T], usize, Option<usize>)> {
        if self.resident {
            if id as usize >= self.resident_centroids.len() {
                return Err(anyhow!("centroid id out of bound access in-memory"));
            }
            return Ok((
                &self.resident_centroids[id as usize].vector,
                self.resident_centroids[id as usize].posting_list_len,
                None,
            ));
        }

        let vector_size = self.num_features * std::mem::size_of::<T>();
        let metadata_size = std::mem::size_of::<usize>();
        let overall_offset = id as usize * (vector_size + metadata_size * 2);
        let file_num = overall_offset / self.backing_file_size;
        if file_num >= self.mmaps.len() {
            return Err(anyhow!(
                "centroid id out of bound: required more files than current mmaps"
            ));
        }

        let file_offset = overall_offset % self.backing_file_size;
        if file_offset >= self.mmaps[file_num].len() {
            return Err(anyhow!("centroid id out of bound access in mmap"));
        }

        let mmap = &self.mmaps[file_num];
        let vector_slice = &mmap[file_offset..file_offset + vector_size];
        let centroid = unsafe {
            std::slice::from_raw_parts(vector_slice.as_ptr() as *const T, self.num_features)
        };
        // TODO(tyb): handle the case where num_features odd -> pad centroid vector
        // (metadata size is either 4 or 8 bytes -> 2 metadata is always 8-byte aligned)
        let len_slice = &mmap[file_offset + vector_size..file_offset + vector_size + metadata_size];
        let posting_list_len = unsafe { *(len_slice.as_ptr() as *const usize) };
        let offset_slice = &mmap[file_offset + vector_size + metadata_size
            ..file_offset + vector_size + metadata_size * 2];
        let posting_list_offset = unsafe { *(offset_slice.as_ptr() as *const usize) };

        Ok((centroid, posting_list_len, Some(posting_list_offset)))
    }

    pub fn append_centroid(&mut self, vector: &[T], posting_list_len: usize) -> Result<()> {
        if vector.len() != self.num_features {
            return Err(anyhow!("vector length mismatch"));
        }

        // TODO(tyb): simplify the code here.
        let size_required =
            self.num_features * std::mem::size_of::<T>() + std::mem::size_of::<usize>();
        let new_size_required = self.size_bytes + size_required;
        let flush = self.resident && new_size_required > self.memory_threshold;

        let centroid = Centroid {
            vector: vector.to_vec(),
            posting_list_len,
        };
        // Good case, where file is still resident
        if self.resident && !flush {
            self.size_bytes = new_size_required;
            self.resident_centroids.push(centroid);
            return Ok(());
        }

        // Spill to disk or create a new file
        if flush {
            self.flush_resident_centroids_to_disk();
        }

        self.append_centroid_to_disk(centroid);
        self.size_bytes = new_size_required;
        Ok(())
    }

    fn flush_resident_posting_lists_to_disk(&mut self) {
        assert!(!self.resident);
        let size_required = std::mem::size_of::<usize>();
        // Extract all the data we need from self to avoid immutable borrow issue
        let posting_lists: Vec<Vec<usize>> = std::mem::take(&mut self.resident_posting_lists);
        for posting_list in posting_lists {
            for idx in posting_list {
                // We have to copy one element of the posting list at a time because we cannot know
                // if the whole posting list will fit into the current mmap.
                if self.current_offset == self.backing_file_size {
                    self.new_backing_file();
                }
                let mut buffer: Vec<u8> = vec![];
                buffer.extend_from_slice(idx.to_le_bytes().as_ref());
                let mmap = &mut self.mmaps[self.current_backing_id as usize];
                // Copy buffer to current offset in mmap
                mmap[self.current_offset..self.current_offset + size_required]
                    .copy_from_slice(&buffer);
                self.current_offset += size_required;
            }
        }
        self.resident_posting_lists.clear();
    }

    fn append_posting_list_to_disk(&mut self, posting_list: &[usize]) {
        let size_required = std::mem::size_of::<usize>();
        for idx in posting_list.iter() {
            // We have to copy one element of the posting list at a time because we cannot know
            // if the whole posting list will fit into the current mmap.
            if self.current_offset == self.backing_file_size {
                self.new_backing_file();
            }
            let mut buffer: Vec<u8> = vec![];
            buffer.extend_from_slice(idx.to_le_bytes().as_ref());
            let mmap = &mut self.mmaps[self.current_backing_id as usize];
            // Copy buffer to current offset in mmap
            mmap[self.current_offset..self.current_offset + size_required].copy_from_slice(&buffer);
            self.current_offset += size_required;
        }
    }

    pub fn get_posting_list(
        &self,
        id: Option<u32>,
        metadata: Option<(usize, usize)>,
    ) -> Result<Cow<[usize]>> {
        if self.resident {
            if id.is_none() {
                return Err(anyhow!("no id to posting list"));
            }
            let i = id.unwrap() as usize;
            if i >= self.resident_posting_lists.len() {
                return Err(anyhow!("posting list id out of bound access in-memory"));
            }
            return Ok(Cow::Borrowed(&self.resident_posting_lists[i]));
        }

        if metadata.is_none() {
            return Err(anyhow!("no metadata for posting list"));
        }

        let (len, offset) = metadata.unwrap();

        let file_num = offset / self.backing_file_size;
        if file_num >= self.mmaps.len() {
            return Err(anyhow!(
                "posting list id out of bound: required more files than current mmaps"
            ));
        }

        let file_offset = offset % self.backing_file_size;
        if file_offset >= self.mmaps[file_num].len() {
            return Err(anyhow!("posting list id out of bound access in mmap"));
        }

        let usize_in_bytes = std::mem::size_of::<usize>();
        let mmap = &self.mmaps[file_num];
        let required_size = len * usize_in_bytes;
        // Posting list fits within a single mmap
        if file_offset + required_size <= mmap.len() {
            let slice = &mmap[file_offset..file_offset + required_size];
            return Ok(Cow::Borrowed(unsafe {
                std::slice::from_raw_parts(slice.as_ptr() as *const usize, len)
            }));
        }

        // Posting list spans across multiple mmaps.
        let mut posting_list = Vec::with_capacity(len);
        let mut remaining_elem = len;
        let mut current_file_num = file_num;
        let mut current_offset = file_offset;
        while remaining_elem > 0 {
            let mmap = &self.mmaps[current_file_num];
            let bytes_left_in_mmap = mmap.len() - current_offset;
            let elems_in_mmap = std::cmp::min(remaining_elem, bytes_left_in_mmap / usize_in_bytes);

            let slice = &mmap[current_offset..current_offset + elems_in_mmap * usize_in_bytes];
            posting_list.extend_from_slice(unsafe {
                std::slice::from_raw_parts(slice.as_ptr() as *const usize, elems_in_mmap)
            });

            remaining_elem -= elems_in_mmap;

            if remaining_elem > 0 {
                current_file_num += 1;
                current_offset = 0;
                if current_file_num >= self.mmaps.len() {
                    return Err(anyhow!(
                        "posting list id spans across multiple mmaps and is out of bound"
                    ));
                }
            }
        }

        Ok(Cow::Owned(posting_list))
    }

    pub fn append_posting_list(&mut self, posting_list: &[usize]) -> Result<()> {
        // TODO(tyb): simplify the code here.
        let size_required = posting_list.len() * std::mem::size_of::<usize>();
        let new_size_required = self.size_bytes + size_required;
        let flush = self.resident && new_size_required > self.memory_threshold;

        // Good case, where file is still resident
        if self.resident && !flush {
            self.size_bytes = new_size_required;
            self.resident_posting_lists.push(posting_list.to_vec());
            return Ok(());
        }

        // Spill to disk or create a new file
        if flush {
            self.flush_resident_centroids_to_disk();
            self.flush_resident_posting_lists_to_disk();
        }

        self.append_posting_list_to_disk(posting_list);
        self.size_bytes = new_size_required;
        Ok(())
    }

    pub fn write(&mut self, writer: &mut std::io::BufWriter<&mut std::fs::File>) -> Result<usize> {
        let mut total_bytes_written = 0;

        // If the data is still resident in memory, flush it to disk first
        if self.resident {
            self.flush_resident_centroids_to_disk();
            self.flush_resident_posting_lists_to_disk();
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

    pub fn config(&self) -> InvertedListStorageConfig {
        InvertedListStorageConfig {
            memory_threshold: self.memory_threshold,
            file_size: self.backing_file_size,
            num_features: self.num_features,
            num_clusters: self.num_clusters,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    #[test]
    fn test_append_centroid_in_memory() {
        let tempdir = tempdir::TempDir::new("append_centroid_in_memory_test").unwrap();
        let base_directory = tempdir.path().to_str().unwrap().to_string();
        let mut storage: FileBackedAppendableInvertedListStorage<f32> =
            FileBackedAppendableInvertedListStorage::new(
                base_directory,
                1024, // memory_threshold
                4096, // backing_file_size
                3,    // num_features
                10,   // num_clusters
            );
        let vector = vec![1.0, 2.0, 3.0];
        storage.append_centroid(&vector, 5).unwrap();

        assert!(storage.resident);
        assert_eq!(storage.resident_centroids.len(), 1);
        assert_eq!(storage.resident_centroids[0].vector, vector);
        assert_eq!(storage.resident_centroids[0].posting_list_len, 5);
    }

    #[test]
    fn test_append_centroid_to_disk() {
        let tempdir = tempdir::TempDir::new("append_centroid_to_disk_test").unwrap();
        let base_directory = tempdir.path().to_str().unwrap().to_string();
        let mut storage: FileBackedAppendableInvertedListStorage<f32> =
            FileBackedAppendableInvertedListStorage::new(
                base_directory,
                1024, // memory_threshold
                4096, // backing_file_size
                3,    // num_features
                10,   // num_clusters
            );
        storage.memory_threshold = 0; // Force writing to disk

        let vector1 = vec![1.0, 2.0, 3.0];
        let vector2 = vec![4.0, 5.0, 6.0];
        storage.append_centroid(&vector1, 5).unwrap();
        storage.append_centroid(&vector2, 7).unwrap();

        assert!(!storage.resident);
        assert_eq!(storage.resident_centroids.len(), 0);
        assert_eq!(storage.mmaps.len(), 1);
    }

    #[test]
    fn test_get_centroid_in_memory() {
        let tempdir = tempdir::TempDir::new("get_centroid_in_memory_test").unwrap();
        let base_directory = tempdir.path().to_str().unwrap().to_string();
        let mut storage: FileBackedAppendableInvertedListStorage<f32> =
            FileBackedAppendableInvertedListStorage::new(
                base_directory,
                1024, // memory_threshold
                4096, // backing_file_size
                3,    // num_features
                10,   // num_clusters
            );
        let vector = vec![1.0, 2.0, 3.0];
        storage.append_centroid(&vector, 5).unwrap();

        let (retrieved_vector, posting_list_len, offset) = storage.get_centroid(0).unwrap();
        assert_eq!(retrieved_vector, vector.as_slice());
        assert_eq!(posting_list_len, 5);
        assert_eq!(offset, None);
    }

    #[test]
    fn test_get_centroid_from_disk() {
        let tempdir = tempdir::TempDir::new("get_centroid_from_disk_test").unwrap();
        let base_directory = tempdir.path().to_str().unwrap().to_string();
        let mut storage: FileBackedAppendableInvertedListStorage<f32> =
            FileBackedAppendableInvertedListStorage::new(
                base_directory,
                1024, // memory_threshold
                4096, // backing_file_size
                3,    // num_features
                10,   // num_clusters
            );
        storage.memory_threshold = 0; // Force writing to disk

        let vector1 = vec![1.0, 2.0, 3.0];
        let vector2 = vec![4.0, 5.0, 6.0];
        storage.append_centroid(&vector1, 5).unwrap();
        storage.append_centroid(&vector2, 7).unwrap();

        let (retrieved_vector, posting_list_len, offset) = storage.get_centroid(1).unwrap();
        assert_eq!(retrieved_vector, vector2.as_slice());
        assert_eq!(posting_list_len, 7);
        assert!(offset.is_some());
    }

    #[test]
    fn test_flush_resident_centroids_to_disk() {
        let tempdir = tempdir::TempDir::new("flush_resident_centroids_to_disk_test").unwrap();
        let base_directory = tempdir.path().to_str().unwrap().to_string();
        let mut storage: FileBackedAppendableInvertedListStorage<f32> =
            FileBackedAppendableInvertedListStorage::new(
                base_directory,
                1024, // memory_threshold
                4096, // backing_file_size
                4,    // num_features (TODO(tyb): switch back to 3 after implementing padding)
                10,   // num_clusters
            );
        let vector1 = vec![1.0, 2.0, 3.0, 4.0];
        let vector2 = vec![5.0, 6.0, 7.0, 8.0];
        storage.append_centroid(&vector1, 5).unwrap();
        storage.append_centroid(&vector2, 7).unwrap();

        assert!(storage.resident);
        storage.flush_resident_centroids_to_disk();
        assert!(!storage.resident);
        assert_eq!(storage.resident_centroids.len(), 0);
        assert_eq!(storage.mmaps.len(), 1);

        // Verify we can still retrieve the centroids
        let (retrieved_vector1, posting_list_len1, _) = storage.get_centroid(0).unwrap();
        let (retrieved_vector2, posting_list_len2, _) = storage.get_centroid(1).unwrap();
        assert_eq!(retrieved_vector1, vector1.as_slice());
        assert_eq!(posting_list_len1, 5);
        assert_eq!(retrieved_vector2, vector2.as_slice());
        assert_eq!(posting_list_len2, 7);
    }

    #[test]
    #[should_panic(expected = "vector length mismatch")]
    fn test_append_centroid_wrong_size() {
        let tempdir = tempdir::TempDir::new("append_centroid_wrong_size_test").unwrap();
        let base_directory = tempdir.path().to_str().unwrap().to_string();
        let mut storage: FileBackedAppendableInvertedListStorage<f32> =
            FileBackedAppendableInvertedListStorage::new(
                base_directory,
                1024, // memory_threshold
                4096, // backing_file_size
                3,    // num_features
                10,   // num_clusters
            );
        let vector = vec![1.0, 2.0]; // Wrong size (should be 3)
        storage.append_centroid(&vector, 5).unwrap();
    }

    #[test]
    #[should_panic(expected = "centroid id out of bound")]
    fn test_get_centroid_out_of_bounds() {
        let tempdir = tempdir::TempDir::new("get_centroid_out_of_bounds_test").unwrap();
        let base_directory = tempdir.path().to_str().unwrap().to_string();
        let mut storage: FileBackedAppendableInvertedListStorage<f32> =
            FileBackedAppendableInvertedListStorage::new(
                base_directory,
                1024, // memory_threshold
                4096, // backing_file_size
                3,    // num_features
                10,   // num_clusters
            );
        storage.get_centroid(0).unwrap(); // No centroids added yet
    }

    #[test]
    fn test_multiple_backing_files() {
        let tempdir = tempdir::TempDir::new("multiple_backing_files_test").unwrap();
        let base_directory = tempdir.path().to_str().unwrap().to_string();
        let mut storage: FileBackedAppendableInvertedListStorage<f32> =
            FileBackedAppendableInvertedListStorage::new(
                base_directory,
                1024, // memory_threshold
                100,  // small backing file size to force multiple files
                4,    // num_features
                10,   // num_clusters
            );
        storage.memory_threshold = 0; // Force writing to disk

        for i in 0..50 {
            let vector = vec![i as f32, (i + 1) as f32, (i + 2) as f32, (i + 3) as f32];
            storage.append_centroid(&vector, i).unwrap();
        }

        assert!(storage.mmaps.len() > 1);

        // Verify we can retrieve centroids from different backing files
        for i in 0..50 {
            let (retrieved_vector, posting_list_len, _) = storage.get_centroid(i).unwrap();
            assert_eq!(
                retrieved_vector,
                &[i as f32, (i + 1) as f32, (i + 2) as f32, (i + 3) as f32]
            );
            assert_eq!(posting_list_len, i as usize);
        }
    }

    #[test]
    fn test_append_posting_list_in_memory() {
        let tempdir = tempdir::TempDir::new("append_posting_list_in_memory_test").unwrap();
        let base_directory = tempdir.path().to_str().unwrap().to_string();
        let mut storage: FileBackedAppendableInvertedListStorage<usize> =
            FileBackedAppendableInvertedListStorage::new(
                base_directory,
                1024, // memory_threshold
                4096, // backing_file_size
                0,    // num_features
                0,    // num_clusters
            );

        let posting_list = vec![1, 2, 3];
        assert!(storage.append_posting_list(&posting_list).is_ok());
        assert_eq!(storage.resident_posting_lists.len(), 1);
        assert_eq!(storage.resident_posting_lists[0], posting_list);
        assert_eq!(
            storage.size_bytes,
            posting_list.len() * std::mem::size_of::<usize>()
        );
    }

    #[test]
    fn test_flush_resident_posting_lists_to_disk() {
        let tempdir = tempdir::TempDir::new("flush_resident_posting_lists_to_disk_test").unwrap();
        let base_directory = tempdir.path().to_str().unwrap().to_string();
        let mut storage = FileBackedAppendableInvertedListStorage::<usize> {
            memory_threshold: 10,
            backing_file_size: 4096,
            num_features: 0,
            num_clusters: 0,
            resident_centroids: vec![],
            resident_posting_lists: vec![vec![1, 2, 3]],
            resident: true,
            size_bytes: 0,
            base_directory,
            mmaps: vec![memmap2::MmapMut::map_anon(4096).unwrap()],
            current_backing_id: 0,
            current_offset: 0,
            current_posting_list_offset: 0,
        };

        // Flush the resident posting lists to disk
        storage.resident = false;
        storage.flush_resident_posting_lists_to_disk();

        // Verify that the data has been written correctly
        let mmap = &storage.mmaps[storage.current_backing_id as usize];
        assert_eq!(
            mmap[0..24],
            [1u8, 0, 0, 0, 0, 0, 0, 0, 2u8, 0, 0, 0, 0, 0, 0, 0, 3u8, 0, 0, 0, 0, 0, 0, 0]
        );
    }

    #[test]
    fn test_get_posting_list_in_memory() {
        let tempdir = tempdir::TempDir::new("get_posting_list_in_memory_test").unwrap();
        let base_directory = tempdir.path().to_str().unwrap().to_string();
        let mut storage = FileBackedAppendableInvertedListStorage::<usize> {
            memory_threshold: 1024,
            backing_file_size: 4096,
            num_features: 0,
            num_clusters: 0,
            resident_centroids: vec![],
            resident_posting_lists: vec![vec![1, 2, 3]],
            resident: true,
            size_bytes: 12,
            base_directory,
            mmaps: vec![],
            current_backing_id: -1,
            current_offset: 0,
            current_posting_list_offset: 0,
        };

        let result = storage.get_posting_list(Some(0), None);
        assert!(result.is_ok());

        let posting_list = result.unwrap();
        assert_eq!(posting_list.as_ref(), &[1, 2, 3]);
    }

    #[test]
    fn test_get_posting_list_from_disk() {
        let tempdir = tempdir::TempDir::new("get_posting_list_from_disk_test").unwrap();
        let base_directory = tempdir.path().to_str().unwrap().to_string();
        let mut storage = FileBackedAppendableInvertedListStorage::<usize> {
            memory_threshold: 10,
            backing_file_size: 4096,
            num_features: 0,
            num_clusters: 0,
            resident_centroids: vec![],
            resident_posting_lists: vec![],
            resident: false, // Start as not resident
            size_bytes: 0,
            base_directory,
            mmaps: vec![memmap2::MmapMut::map_anon(4096).unwrap()],
            current_backing_id: 0,
            current_offset: 0,
            current_posting_list_offset: 0,
        };

        // Append a posting list and flush it to disk
        let posting_list = vec![4, 5, 6];
        storage.append_posting_list(&posting_list);

        // Retrieve the posting list from disk using its metadata (length and offset)
        let metadata = Some((posting_list.len(), 0));

        let result = storage.get_posting_list(None, metadata);

        assert!(result.is_ok());

        let retrieved_posting_list = result.unwrap();

        // Verify that the retrieved posting list matches what was stored
        assert_eq!(retrieved_posting_list.as_ref(), &[4, 5, 6]);
    }

    #[test]
    fn test_get_posting_list_spanning_multiple_mmaps() {
        let tempdir =
            tempdir::TempDir::new("get_posting_list_spanning_multiple_mmaps_test").unwrap();
        let base_directory = tempdir.path().to_str().unwrap().to_string();

        // Calculate a small backing file size that will force spanning
        let usize_size = std::mem::size_of::<usize>();
        let backing_file_size = usize_size * 3; // Each mmap can hold 3 usize elements

        // Initialize the storage with small backing file size
        let mut storage = FileBackedAppendableInvertedListStorage::<usize> {
            memory_threshold: 10,
            backing_file_size,
            num_features: 0,
            num_clusters: 0,
            resident_centroids: vec![],
            resident_posting_lists: vec![],
            resident: false, // Start as not resident
            size_bytes: 0,
            base_directory: base_directory.clone(),
            mmaps: vec![],
            current_backing_id: -1,
            current_offset: 0,
            current_posting_list_offset: 0,
        };

        // Create a large posting list that will span multiple mmaps
        let large_posting_list: Vec<usize> = (1..10).collect(); // 9 elements

        // Manually simulate appending the large posting list
        // This part would normally be done by append_posting_list, but we're doing it manually for testing
        let mut current_backing_id = storage.current_backing_id;
        let mut current_offset = backing_file_size;
        for &value in &large_posting_list {
            if current_offset == backing_file_size {
                // Simulate creating a new backing file
                current_backing_id += 1;
                current_offset = 0;
                storage
                    .mmaps
                    .push(memmap2::MmapMut::map_anon(backing_file_size).unwrap());
            }

            let mmap = &mut storage.mmaps[current_backing_id as usize];
            mmap[current_offset..current_offset + usize_size].copy_from_slice(&value.to_le_bytes());
            current_offset += usize_size;
        }

        // Set the final state of the storage
        storage.current_backing_id = current_backing_id as i32;
        storage.current_offset = current_offset;

        // Calculate the total size of the posting list in bytes
        let total_size = large_posting_list.len() * usize_size;

        // Retrieve the posting list
        let result = storage.get_posting_list(None, Some((large_posting_list.len(), 0)));

        assert!(result.is_ok());

        let retrieved_posting_list = result.unwrap();

        // Verify that the retrieved posting list matches the original
        assert_eq!(
            retrieved_posting_list.as_ref(),
            large_posting_list.as_slice()
        );

        // Verify that the posting list indeed spans multiple mmaps
        assert_eq!(
            storage.mmaps.len(),
            (total_size + backing_file_size - 1) / backing_file_size
        );
    }
}
