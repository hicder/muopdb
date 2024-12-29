use std::sync::Arc;

use index::collection::reader::Reader;
use index::collection::Collection;

pub struct IndexProvider {
    data_directory: String,
}

impl IndexProvider {
    pub fn new(data_directory: String) -> Self {
        Self { data_directory }
    }

    pub fn read_index(&self, name: &str) -> Option<Arc<Collection>> {
        let index_path = format!("{}/{}", self.data_directory, name);
        let reader = Reader::new(index_path);
        match reader.read() {
            Ok(collection) => Some(collection),
            Err(e) => {
                println!("Failed to read collection: {}", e);
                None
            }
        }
    }
}
