use std::collections::HashMap;
use index::collection::BoxedCollection;

pub struct CollectionCatalog {
    collections: HashMap<String, BoxedCollection>,
}

impl CollectionCatalog {
    pub fn new() -> Self {
        Self {
            collections: HashMap::new(),
        }
    }

    pub async fn add_collection(&mut self, name: String, collection: BoxedCollection) {
        self.collections.insert(name, collection);
    }

    pub async fn get_collection(&self, name: &str) -> Option<BoxedCollection> {
        self.collections
            .get(name)
            .map(|collection| (*collection).clone())
    }

    pub async fn get_all_collection_names_sorted(&self) -> Vec<String> {
        let mut v: Vec<String> = self.collections.keys().cloned().collect();
        v.sort();
        v
    }

    pub async fn collection_exists(&self, name: &str) -> bool {
        self.collections.contains_key(name)
    }
}
