use std::collections::HashMap;
use std::sync::Arc;

use index::index::BoxedIndex;

pub struct IndexCatalog {
    indexes: HashMap<String, Arc<BoxedIndex>>,
}

impl IndexCatalog {
    pub fn new() -> Self {
        Self {
            indexes: HashMap::new(),
        }
    }

    pub async fn add_index(&mut self, name: String, index: Arc<BoxedIndex>) {
        self.indexes.insert(name, index);
    }

    pub async fn get_index(&self, name: &str) -> Option<Arc<BoxedIndex>> {
        self.indexes.get(name).map(|index| index.clone())
    }

    pub async fn get_all_index_names_sorted(&self) -> Vec<String> {
        let mut v: Vec<String> = self.indexes.keys().cloned().collect();
        v.sort();
        v
    }
}
