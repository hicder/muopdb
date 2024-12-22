use std::collections::HashMap;
use std::sync::Arc;

use index::index::BoxedSearchable;

pub struct IndexCatalog {
    indexes: HashMap<String, Arc<BoxedSearchable>>,
}

impl IndexCatalog {
    pub fn new() -> Self {
        Self {
            indexes: HashMap::new(),
        }
    }

    pub async fn add_index(&mut self, name: String, index: Arc<BoxedSearchable>) {
        self.indexes.insert(name, index);
    }

    pub async fn get_index(&self, name: &str) -> Option<Arc<BoxedSearchable>> {
        self.indexes.get(name).map(|index| index.clone())
    }

    pub async fn get_all_index_names_sorted(&self) -> Vec<String> {
        let mut v: Vec<String> = self.indexes.keys().cloned().collect();
        v.sort();
        v
    }
}
