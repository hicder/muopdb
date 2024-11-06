use std::marker::PhantomData;

use anyhow::Result;
use memmap2::Mmap;
use num_traits::ToBytes;
use utils::mem::transmute_u8_to_slice;

use crate::hnsw::utils::TraversalContext;
use crate::utils::SearchContext;

pub struct FixedFileVectorStorage<T> {
    _marker: PhantomData<T>,

    mmaps: Mmap,
    num_features: usize,
    file_path: String,
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
            file_path,
        })
    }

    pub fn get(&self, index: usize, context: &mut SearchContext) -> Option<&[T]> {
        let start = 8 + index * self.num_features * std::mem::size_of::<T>();
        let end = 8 + (index + 1) * self.num_features * std::mem::size_of::<T>();
        if end > self.mmaps.len() {
            return None;
        }

        if context.should_record_pages() {
            let page_id = format!("{}::{}", self.file_path, self.get_page_id(start));
            context.record_pages(page_id);
        }

        let slice = &self.mmaps[start..end];
        Some(transmute_u8_to_slice::<T>(slice))
    }

    fn get_page_id(&self, index: usize) -> usize {
        index / 4096
    }
}

// Test
#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::BufWriter;

    use super::*;
    use crate::vector::file::FileBackedAppendableVectorStorage;
    use crate::vector::VectorStorage;

    #[test]
    fn test_fixed_file_vector_storage() {
        let tempdir = tempdir::TempDir::new("vector_storage_test").unwrap();
        let base_directory = tempdir.path().to_str().unwrap().to_string();
        let mut appendable_storage =
            FileBackedAppendableVectorStorage::<u32>::new(base_directory.clone(), 4, 8192, 4);
        for i in 0..257 {
            appendable_storage.append(&vec![i, i, i, i]).unwrap();
        }

        let vectors_path = format!("{}/vector_storage", base_directory);
        let mut vectors_file = File::create(vectors_path.clone()).unwrap();
        let mut vectors_buffer_writer = BufWriter::new(&mut vectors_file);

        appendable_storage
            .write(&mut vectors_buffer_writer)
            .unwrap();

        let mut context = SearchContext::new(true);
        let storage = FixedFileVectorStorage::<u32>::new(vectors_path, 4).unwrap();
        assert_eq!(storage.get(0, &mut context).unwrap(), &[0, 0, 0, 0]);
        assert_eq!(
            storage.get(256, &mut context).unwrap(),
            &[256, 256, 256, 256]
        );
        assert_eq!(context.num_pages_accessed(), 2);
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

        let vectors_path = format!("{}/vector_storage", base_directory);
        let mut vectors_file = File::create(vectors_path.clone()).unwrap();
        let mut vectors_buffer_writer = BufWriter::new(&mut vectors_file);

        appendable_storage
            .write(&mut vectors_buffer_writer)
            .unwrap();

        let mut context = SearchContext::new(false);
        let storage = FixedFileVectorStorage::<f32>::new(vectors_path, 4).unwrap();
        assert_eq!(storage.get(0, &mut context).unwrap(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(storage.get(1, &mut context).unwrap(), &[5.0, 6.0, 7.0, 8.0]);
        assert_eq!(
            storage.get(2, &mut context).unwrap(),
            &[9.0, 10.0, 11.0, 12.0]
        );
    }
}
