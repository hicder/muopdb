use std::collections::HashMap;
use std::sync::Arc;

use index::collection::Collection;

pub struct IndexCatalog {
    collections: HashMap<String, Arc<Collection>>,
}

impl IndexCatalog {
    pub fn new() -> Self {
        Self {
            collections: HashMap::new(),
        }
    }

    pub async fn add_collection(&mut self, name: String, collection: Arc<Collection>) {
        self.collections.insert(name, collection);
    }

    pub async fn get_collection(&self, name: &str) -> Option<Arc<Collection>> {
        self.collections
            .get(name)
            .map(|collection| collection.clone())
    }

    pub async fn get_all_collection_names_sorted(&self) -> Vec<String> {
        let mut v: Vec<String> = self.collections.keys().cloned().collect();
        v.sort();
        v
    }
}
