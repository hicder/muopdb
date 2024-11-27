use std::marker::PhantomData;

use anyhow::{anyhow, Result};
use memmap2::Mmap;
use utils::mem::transmute_u8_to_slice;

const PL_METADATA_LEN: usize = 2;

pub struct FixedFilePostingListStorage {
    _marker: PhantomData<u64>,

    mmap: Mmap,
    pub num_clusters: usize,
}

impl FixedFilePostingListStorage {
    pub fn new(file_path: String) -> Result<Self> {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .open(file_path.clone())?;
        let mmap = unsafe { Mmap::map(&file) }?;
        let num_clusters = usize::from_le_bytes(mmap[0..8].try_into()?);
        Ok(Self {
            _marker: PhantomData,
            mmap,
            num_clusters,
        })
    }

    pub fn get(&self, index: usize) -> Result<&[u64]> {
        if index >= self.num_clusters {
            return Err(anyhow!("Index out of bound"));
        }

        let size_in_bytes = std::mem::size_of::<u64>();
        // Data start at offset 8, since the first u64 is num_clusters
        let data_offset = 8;
        let metadata_offset = data_offset + index * PL_METADATA_LEN * size_in_bytes;

        let slice = &self.mmap[metadata_offset..metadata_offset + size_in_bytes];
        let pl_len = u64::from_le_bytes(slice.try_into()?) as usize;
        let slice = &self.mmap
            [metadata_offset + size_in_bytes..metadata_offset + PL_METADATA_LEN * size_in_bytes];
        let pl_offset = u64::from_le_bytes(slice.try_into()?) as usize + data_offset;

        let slice = &self.mmap[pl_offset..pl_offset + pl_len * std::mem::size_of::<u64>()];
        Ok(transmute_u8_to_slice::<u64>(slice))
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::BufWriter;

    use super::*;
    use crate::posting_list::file::FileBackedAppendablePostingListStorage;
    use crate::posting_list::PostingListStorage;

    #[test]
    fn test_fixed_file_posting_list_storage() {
        // Create a temporary directory for our test file
        let tempdir = tempdir::TempDir::new("fixed_file_posting_list_storage_test").unwrap();
        let base_directory = tempdir.path().to_str().unwrap().to_string();
        let mut appendable_storage =
            FileBackedAppendablePostingListStorage::new(base_directory.clone(), 1024, 4096, 3);
        appendable_storage
            .append(&vec![1, 2, 3, 4])
            .expect("Failed to append posting list");
        appendable_storage
            .append(&vec![5, 6, 7, 8])
            .expect("Failed to append posting list");
        appendable_storage
            .append(&vec![9, 10, 11, 12])
            .expect("Failed to append posting list");

        let vectors_path = format!("{}/vector_storage", base_directory);
        let mut vectors_file = File::create(vectors_path.clone()).unwrap();
        let mut vectors_buffer_writer = BufWriter::new(&mut vectors_file);

        appendable_storage
            .write(&mut vectors_buffer_writer)
            .expect("Failed to write posting list to file");

        let storage = FixedFilePostingListStorage::new(vectors_path)
            .expect("Failed to create fixed posting list storage");
        assert_eq!(storage.num_clusters, 3);
        assert_eq!(
            storage.get(0).expect("Failed to read back posting list"),
            &[1, 2, 3, 4]
        );
        assert_eq!(
            storage.get(1).expect("Failed to read back posting list"),
            &[5, 6, 7, 8]
        );
        assert_eq!(
            storage.get(2).expect("Failed to read back posting list"),
            &[9, 10, 11, 12]
        );

        // Test out of bounds access
        assert!(storage.get(3).is_err());
    }
}
