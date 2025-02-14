use std::fs::OpenOptions;
use std::io::Write;
use std::vec;

use anyhow::{anyhow, Result};
use num_traits::ToBytes;
use utils::io::wrap_write;

use super::{StorageContext, VectorStorageConfig};

pub struct FileBackedAppendableVectorStorage<T: ToBytes + Clone> {
    pub memory_threshold: usize,
    pub backing_file_size: usize,
    num_features: usize,

    // Whether it's currently in memory
    resident_vectors: Vec<Vec<T>>,
    resident: bool,
    size_bytes: usize,

    // Only has value if we spill to disk
    base_directory: String,
    mmaps: Vec<memmap2::MmapMut>,
    current_backing_id: i32,
    current_offset: usize,
}

impl<T: ToBytes + Clone> FileBackedAppendableVectorStorage<T> {
    pub fn new(
        base_directory: String,
        memory_threshold: usize,
        backing_file_size: usize,
        num_features: usize,
    ) -> Self {
        let bytes_per_vector = num_features * std::mem::size_of::<T>();
        let rounded_backing_file_size = backing_file_size / bytes_per_vector * bytes_per_vector;
        Self {
            base_directory,
            memory_threshold,
            backing_file_size: rounded_backing_file_size,
            num_features,
            resident_vectors: vec![],
            resident: true,
            size_bytes: 0,
            mmaps: vec![],
            current_backing_id: -1,
            current_offset: 0,
        }
    }

    pub fn new_with_config(base_directory: String, config: VectorStorageConfig) -> Self {
        Self::new(
            base_directory,
            config.memory_threshold,
            config.file_size,
            config.num_features,
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

    // Caller is responsible for preparing new mmaps if necessary.
    fn append_resident_to_current_mmap(&mut self, resident_idx: usize) {
        let vector = &self.resident_vectors[resident_idx];
        let size_required = vector.len() * std::mem::size_of::<T>();
        let mut buffer: Vec<u8> = vec![];
        for i in 0..vector.len() {
            buffer.extend_from_slice(vector[i].to_le_bytes().as_ref());
        }

        let mmap = &mut self.mmaps[self.current_backing_id as usize];
        // Copy buffer to current offset in mmap
        mmap[self.current_offset..self.current_offset + size_required].copy_from_slice(&buffer);
        self.current_offset += size_required;
    }

    fn new_backing_file(&mut self) -> Result<()> {
        self.current_backing_id += 1;
        let backing_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(format!(
                "{}/vector.bin.{}",
                self.base_directory, self.current_backing_id
            ))?;

        backing_file.set_len(self.backing_file_size as u64)?;

        self.mmaps
            .push(unsafe { memmap2::MmapMut::map_mut(&backing_file)? });
        self.current_offset = 0;
        Ok(())
    }

    fn flush_resident_to_disk(&mut self) -> Result<()> {
        let len = self.resident_vectors.len();
        for i in 0..len {
            if self.current_offset == self.backing_file_size {
                self.new_backing_file()?;
            }
            self.append_resident_to_current_mmap(i);
        }
        self.resident_vectors.clear();
        Ok(())
    }

    fn append_vector_to_disk(&mut self, vector: &[T]) -> Result<()> {
        if self.current_offset == self.backing_file_size {
            self.new_backing_file()?;
        }
        if self.current_offset + vector.len() * std::mem::size_of::<T>() > self.backing_file_size {
            return Err(anyhow!("vector too big to be flushed to backing file"));
        }

        let mmap = &mut self.mmaps[self.current_backing_id as usize];
        vector.iter().for_each(|v| {
            let item_bytes = v.to_le_bytes();
            let bytes = item_bytes.as_ref();
            mmap[self.current_offset..self.current_offset + bytes.len()].copy_from_slice(bytes);
            self.current_offset += bytes.len();
        });

        Ok(())
    }
}

impl<T: ToBytes + Clone> FileBackedAppendableVectorStorage<T> {
    pub fn get_no_context(&self, id: u32) -> Result<&[T]> {
        if self.resident {
            if id as usize >= self.resident_vectors.len() {
                return Err(anyhow!("vector id out of bound"));
            }
            return Ok(&self.resident_vectors[id as usize]);
        }

        let overall_offset = id as usize * self.num_features * std::mem::size_of::<T>();
        let file_num = overall_offset / self.backing_file_size;
        if file_num >= self.mmaps.len() {
            return Err(anyhow!("file number out of bound"));
        }

        let file_offset = overall_offset % self.backing_file_size;
        if file_offset >= self.mmaps[file_num].len() {
            return Err(anyhow!("mmap offset out of bound"));
        }

        let mmap = &self.mmaps[file_num];
        let slice = &mmap[file_offset..file_offset + self.num_features * std::mem::size_of::<T>()];
        Ok(unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const T, self.num_features) })
    }

    pub fn get(&self, id: u32, _context: &mut impl StorageContext) -> Result<&[T]> {
        self.get_no_context(id)
    }

    pub fn append(&mut self, vector: &[T]) -> Result<()> {
        if vector.len() != self.num_features {
            return Err(anyhow!(
                "vector length mismatch: expected {}, got {}",
                self.num_features,
                vector.len()
            ));
        }

        let size_required = vector.len() * std::mem::size_of::<T>();
        let new_size_required = self.size_bytes + size_required;
        let current_resident = self.resident;
        if !current_resident || new_size_required > self.memory_threshold {
            self.resident = false;
        }

        // Good case, where file is still resident
        if self.resident {
            self.size_bytes = new_size_required;
            self.resident_vectors.push(vector.to_vec());
            return Ok(());
        }

        // Spill to disk or create a new file
        if current_resident != self.resident {
            self.current_offset = self.backing_file_size;
            self.flush_resident_to_disk()?;
        }

        self.append_vector_to_disk(vector)?;
        self.size_bytes = new_size_required;
        Ok(())
    }

    pub fn num_vectors(&self) -> usize {
        self.size_bytes / (self.num_features * std::mem::size_of::<T>())
    }

    // TODO(hicder): Just copy the backed file to the output file for optimization
    pub fn write(&self, writer: &mut std::io::BufWriter<&mut std::fs::File>) -> Result<usize> {
        let num_vectors = self.num_vectors() as u64;
        let mut len = 0;
        len += wrap_write(writer, &num_vectors.to_le_bytes())?;
        for i in 0..num_vectors {
            let vector = self.get_no_context(i as u32).unwrap();
            for j in 0..self.num_features {
                len += wrap_write(writer, vector[j].to_le_bytes().as_ref())?;
            }
            writer.flush()?;
        }
        Ok(len)
    }

    pub fn config(&self) -> VectorStorageConfig {
        VectorStorageConfig {
            memory_threshold: self.memory_threshold,
            file_size: self.backing_file_size,
            num_features: self.num_features,
        }
    }

    pub fn multi_get(&self, ids: &[u32], _context: &mut impl StorageContext) -> Result<Vec<&[T]>> {
        let mut result = vec![];
        for id in ids {
            result.push(self.get(*id, _context)?);
        }
        Ok(result)
    }
}

// Test
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_file_backed_vector_storage() {
        let tempdir = tempdir::TempDir::new("vector_storage_test").unwrap();
        let base_directory = tempdir.path().to_str().unwrap().to_string();
        let mut storage =
            FileBackedAppendableVectorStorage::<u32>::new(base_directory, 1024, 1024, 4);
        let first_vector = vec![1, 2, 3, 4];
        let second_vector = vec![5, 6, 7, 8];
        for _ in 0..64 {
            storage
                .append(&first_vector)
                .unwrap_or_else(|_| panic!("append failed"));
        }
        assert!(storage.is_resident());

        storage
            .append(&second_vector)
            .unwrap_or_else(|_| panic!("append failed"));
        assert!(!storage.is_resident());
        storage.flush().unwrap_or_else(|_| panic!("flush failed"));

        let vec = storage.get_no_context(0).unwrap();
        assert_eq!(vec[0], 1);
        assert_eq!(vec[1], 2);
        assert_eq!(vec[2], 3);
        assert_eq!(vec[3], 4);

        let vec = storage.get_no_context(64).unwrap();
        assert_eq!(vec[0], 5);
        assert_eq!(vec[1], 6);
        assert_eq!(vec[2], 7);
        assert_eq!(vec[3], 8);
    }

    #[test]
    fn test_file_backed_appendable_vector_storage() {
        // Create a temporary directory for our test
        let tempdir = tempdir::TempDir::new("vector_storage_test").unwrap();
        let base_directory = tempdir.path().to_str().unwrap().to_string();

        // Create a new storage with a small memory threshold to force disk usage
        let mut storage = FileBackedAppendableVectorStorage::<f32>::new(
            base_directory,
            100,  // Small memory threshold
            1024, // Backing file size
            3,    // Number of features
        );

        // Test appending vectors while in memory
        let vector1 = vec![1.0, 2.0, 3.0];
        let vector2 = vec![4.0, 5.0, 6.0];
        storage.append(&vector1).unwrap();
        storage.append(&vector2).unwrap();

        // Check if vectors are correctly stored and retrieved
        assert_eq!(storage.get_no_context(0).unwrap(), &vector1);
        assert_eq!(storage.get_no_context(1).unwrap(), &vector2);
        assert!(storage.is_resident());

        // Append more vectors to force disk usage
        for i in 0..10 {
            let vector = vec![i as f32, (i + 1) as f32, (i + 2) as f32];
            storage.append(&vector).unwrap();
        }

        // Check if storage is no longer resident in memory
        assert!(!storage.is_resident());

        // Verify all vectors are still accessible
        assert_eq!(storage.get_no_context(0).unwrap(), &vector1);
        assert_eq!(storage.get_no_context(1).unwrap(), &vector2);
        for i in 0..10 {
            let expected = vec![i as f32, (i + 1) as f32, (i + 2) as f32];
            assert_eq!(storage.get_no_context((i + 2) as u32).unwrap(), &expected);
        }

        // Test length
        assert_eq!(storage.num_vectors(), 12);

        // Test flush
        assert!(storage.flush().is_ok());

        // Test appending a vector with incorrect length
        let invalid_vector = vec![1.0, 2.0];
        assert!(storage.append(&invalid_vector).is_err());

        // Test getting an out-of-bounds vector
        assert!(storage.get_no_context(100).is_err());
    }
}
