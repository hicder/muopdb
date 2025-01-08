use anyhow::Result;
use memmap2::Mmap;

use crate::multi_spann::index::MultiSpannIndex;

pub struct MultiSpannReader {
    base_directory: String,
}

impl MultiSpannReader {
    pub fn new(base_directory: String) -> Self {
        Self { base_directory }
    }

    pub fn read(&self) -> Result<MultiSpannIndex> {
        let user_index_info_file_path = format!("{}/user_index_info", self.base_directory);
        let user_index_info_file = std::fs::OpenOptions::new()
            .read(true)
            .open(user_index_info_file_path)?;

        let user_index_info_mmap = unsafe { Mmap::map(&user_index_info_file)? };
        MultiSpannIndex::new(self.base_directory.clone(), user_index_info_mmap)
    }
}
