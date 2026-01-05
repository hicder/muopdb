/// Config for vector storage.
pub struct VectorStorageConfig {
    pub memory_threshold: usize,
    pub file_size: usize,
    pub num_features: usize,
}

pub trait StorageContext {
    fn should_record_pages(&self) -> bool;
    fn record_pages(&mut self, page_id: String);
    fn num_pages_accessed(&self) -> usize;
    fn reset_pages_accessed(&mut self);
    fn set_visited(&mut self, id: u32);
    fn visited(&self, id: u32) -> bool;
}

pub mod async_storage;
pub mod file;
