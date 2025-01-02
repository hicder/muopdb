use std::sync::Arc;

use index::collection::reader::Reader;
use index::collection::Collection;

pub struct CollectionProvider {
    data_directory: String,
}

impl CollectionProvider {
    pub fn new(data_directory: String) -> Self {
        Self { data_directory }
    }

    pub fn read_collection(&self, name: &str) -> Option<Arc<Collection>> {
        let collection_path = format!("{}/{}", self.data_directory, name);
        let reader = Reader::new(collection_path);
        match reader.read() {
            Ok(collection) => Some(collection),
            Err(e) => {
                println!("Failed to read collection: {}", e);
                None
            }
        }
    }
}
