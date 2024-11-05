use std::marker::PhantomData;

use anyhow::Result;
use memmap2::Mmap;
use num_traits::ToBytes;
use utils::mem::transmute_u8_to_slice;

pub struct FixedFileVectorStorage<T> {
    _marker: PhantomData<T>,

    mmaps: Mmap,
    num_features: usize,
}

impl<T: ToBytes + Clone> FixedFileVectorStorage<T> {
    pub fn new(file_path: String, num_features: usize) -> Result<Self> {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .open(file_path.clone())?;
        let mmap = unsafe { Mmap::map(&file) }?;
        Ok(Self {
            _marker: PhantomData,
            mmaps: mmap,
            num_features,
        })
    }

    pub fn get(&self, index: usize) -> Option<&[T]> {
        let start = index * self.num_features * std::mem::size_of::<T>();
        let end = (index + 1) * self.num_features * std::mem::size_of::<T>();
        if end > self.mmaps.len() {
            return None;
        }

        let slice = &self.mmaps[start..end];
        Some(transmute_u8_to_slice::<T>(slice))
    }
}

// Test
#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::file::FileBackedAppendableVectorStorage;
    use crate::vector::VectorStorage;

    #[test]
    fn test_fixed_file_vector_storage() {
        let tempdir = tempdir::TempDir::new("vector_storage_test").unwrap();
        let base_directory = tempdir.path().to_str().unwrap().to_string();
        let mut appendable_storage =
            FileBackedAppendableVectorStorage::<u32>::new(base_directory.clone(), 4, 1024, 4);
        appendable_storage.append(&vec![1, 2, 3, 4]).unwrap();
        appendable_storage.append(&vec![5, 6, 7, 8]).unwrap();
        appendable_storage.append(&vec![9, 10, 11, 12]).unwrap();

        let file_path = format!("{}/vector.bin.0", base_directory);

        let storage = FixedFileVectorStorage::<u32>::new(file_path, 4).unwrap();
        assert_eq!(storage.get(0).unwrap(), &[1, 2, 3, 4]);
        assert_eq!(storage.get(1).unwrap(), &[5, 6, 7, 8]);
        assert_eq!(storage.get(2).unwrap(), &[9, 10, 11, 12]);
    }

    #[test]
    fn test_fixed_file_vector_storage_f32() {
        let tempdir = tempdir::TempDir::new("vector_storage_test").unwrap();
        let base_directory = tempdir.path().to_str().unwrap().to_string();
        let mut appendable_storage =
            FileBackedAppendableVectorStorage::<f32>::new(base_directory.clone(), 4, 1024, 4);
        appendable_storage
            .append(&vec![1.0, 2.0, 3.0, 4.0])
            .unwrap();
        appendable_storage
            .append(&vec![5.0, 6.0, 7.0, 8.0])
            .unwrap();
        appendable_storage
            .append(&vec![9.0, 10.0, 11.0, 12.0])
            .unwrap();

        let file_path = format!("{}/vector.bin.0", base_directory);
        let storage = FixedFileVectorStorage::<f32>::new(file_path, 4).unwrap();
        assert_eq!(storage.get(0).unwrap(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(storage.get(1).unwrap(), &[5.0, 6.0, 7.0, 8.0]);
        assert_eq!(storage.get(2).unwrap(), &[9.0, 10.0, 11.0, 12.0]);
    }
}
