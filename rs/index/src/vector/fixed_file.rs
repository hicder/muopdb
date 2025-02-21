use std::collections::HashSet;
use std::marker::PhantomData;
use std::sync::RwLock;
use anyhow::Result;
use memmap2::Mmap;
use num_traits::ToBytes;
use quantization::quantization::Quantizer;
use utils::mem::transmute_u8_to_slice;

use crate::utils::{IntermediateResult, PointAndDistance, SearchContext, SearchStats};
use crate::vector::StorageContext;

pub struct FixedFileVectorStorage<T> {
    _marker: PhantomData<T>,

    mmaps: Mmap,
    pub num_vectors: usize,
    num_features: usize,
    file_path: String,
    offset: usize,
}

impl<T: ToBytes + Clone> FixedFileVectorStorage<T> {
    pub fn new(file_path: String, num_features: usize) -> Result<Self> {
        Self::new_with_offset(file_path, num_features, 0)
    }

    pub fn new_with_offset(file_path: String, num_features: usize, offset: usize) -> Result<Self> {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .open(file_path.clone())?;
        let mmap = unsafe { Mmap::map(&file) }?;
        let num_vectors = usize::from_le_bytes(mmap[offset..offset + 8].try_into().unwrap());
        Ok(Self {
            _marker: PhantomData,
            mmaps: mmap,
            num_vectors,
            num_features,
            file_path,
            offset,
        })
    }

    fn get_page_id(&self, index: usize) -> usize {
        index / 4096
    }

    fn vector_size_in_bytes(num_features: usize) -> usize {
        num_features * std::mem::size_of::<T>()
    }
}

impl<T: ToBytes + Clone> FixedFileVectorStorage<T> {
    pub fn multi_get(&self, ids: &[u32], context: &mut impl StorageContext) -> Result<Vec<&[T]>> {
        let mut result = vec![];
        for id in ids {
            // TODO: Handle error
            result.push(self.get(*id, context).unwrap());
        }
        Ok(result)
    }

    pub fn get_no_context(&self, id: u32) -> Result<&[T]> {
        if id as usize >= self.num_vectors {
            return Err(anyhow::anyhow!("index out of bounds"));
        }
        let start = self.offset + 8 + (id as usize) * Self::vector_size_in_bytes(self.num_features);
        Ok(transmute_u8_to_slice::<T>(
            &self.mmaps[start..start + Self::vector_size_in_bytes(self.num_features)],
        ))
    }

    pub fn get(&self, id: u32, context: &mut impl StorageContext) -> Result<&[T]> {
        if id as usize >= self.num_vectors {
            return Err(anyhow::anyhow!("index out of bounds"));
        }
        let start = self.offset + 8 + (id as usize) * Self::vector_size_in_bytes(self.num_features);

        if context.should_record_pages() {
            let page_id = format!("{}::{}", self.file_path, self.get_page_id(start));
            context.record_pages(page_id);
        }

        let slice = &self.mmaps[start..start + Self::vector_size_in_bytes(self.num_features)];
        Ok(transmute_u8_to_slice::<T>(slice))
    }

    pub async fn get_async(&self, id: u32, context: &mut impl StorageContext) -> Result<&[T]> {
        self.get(id, context)
    }

    pub fn num_vectors(&self) -> usize {
        self.num_vectors
    }

    pub fn config(&self) -> super::VectorStorageConfig {
        super::VectorStorageConfig {
            memory_threshold: 0,
            file_size: 0,
            num_features: self.num_features,
        }
    }

    // This can avoid repeatedly calling `get` by the caller, reducing vtable lookup
    // cost if the caller is an instance of Box<VectorStorage>
    pub async fn compute_distance_batch_async(
        &self,
        query: &[T],
        iterator: impl Iterator<Item = u64>,
        quantizer: &impl Quantizer<QuantizedT = T>,
        invalidated_ids: &RwLock<HashSet<u32>>,
        record_pages: bool,
    ) -> Result<IntermediateResult> {
        let mut result = vec![];
        let mut context = SearchContext::new(record_pages);
        let mut stats = SearchStats::new();
        let invalidate_ids_clone = {
            let guard = invalidated_ids.read().unwrap();
            if guard.is_empty() {
                None
            } else {
                Some(guard.clone())
            }
        };

        for id in iterator {
            // Skip invalidated ids
            if invalidate_ids_clone.as_ref().map_or(
                false,
                |ids| ids.contains(&(id as u32))
            ) {
                continue;
            }

            let vector = self.get_async(id as u32, &mut context).await?;
            let distance = quantizer.distance(
                query,
                vector,
                utils::distance::l2::L2DistanceCalculatorImpl::StreamingSIMD,
            );
            result.push(PointAndDistance::new(distance, id as u32));
        }
        stats.num_pages_accessed = context.num_pages_accessed();
        Ok(IntermediateResult {
            point_and_distances: result,
            stats,
        })
    }
}

// Test
#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::BufWriter;

    use super::*;
    use crate::utils::SearchContext;
    use crate::vector::file::FileBackedAppendableVectorStorage;

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
        assert_eq!(storage.num_vectors, 3);
        assert_eq!(storage.get(0, &mut context).unwrap(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(storage.get(1, &mut context).unwrap(), &[5.0, 6.0, 7.0, 8.0]);
        assert_eq!(
            storage.get(2, &mut context).unwrap(),
            &[9.0, 10.0, 11.0, 12.0]
        );

        // Test out of bounds access
        assert!(storage.get(3, &mut context).is_err());
    }

    #[test]
    fn test_vector_size_in_bytes() {
        assert_eq!(FixedFileVectorStorage::<f32>::vector_size_in_bytes(3), 12); // 3 features * 4 bytes (size of f32)
        assert_eq!(FixedFileVectorStorage::<u64>::vector_size_in_bytes(3), 24); // 3 features * 8 bytes (size of u64)
        assert_eq!(FixedFileVectorStorage::<u8>::vector_size_in_bytes(4), 4); // 4 features * 1 byte (size of u8)
        assert_eq!(FixedFileVectorStorage::<u16>::vector_size_in_bytes(4), 8); // 4 features * 2 bytes (size of u16)
    }
}
