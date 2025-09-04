use std::collections::HashMap;

use index::collection::BoxedCollection;
use metrics::INTERNAL_METRICS;

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
        INTERNAL_METRICS.num_collections.inc();
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use config::collection::CollectionConfig;
    use index::collection::core::Collection;
    use metrics::INTERNAL_METRICS;
    use quantization::noq::noq::NoQuantizer;
    use tempdir::TempDir;
    use utils::distance::l2::L2DistanceCalculator;

    use super::*;

    #[tokio::test]
    async fn test_collection_catalog_metrics() {
        // Get initial metric value
        let initial_count = INTERNAL_METRICS.num_collections.get();

        // Create a new catalog
        let mut catalog = CollectionCatalog::new();

        let collection_name = "test_num_collections_metrics";
        let temp_dir = TempDir::new(collection_name).unwrap();
        let base_directory: String = temp_dir.path().to_str().unwrap().to_string();

        // Create and add a mock collection
        let collection: BoxedCollection = BoxedCollection::CollectionNoQuantizationL2(Arc::new(
            Collection::<NoQuantizer<L2DistanceCalculator>>::new(
                collection_name.to_string(),
                base_directory,
                CollectionConfig::default(),
            )
            .unwrap(),
        ));
        catalog
            .add_collection("test1".to_string(), collection.clone())
            .await;

        // Verify metric was incremented
        assert_eq!(INTERNAL_METRICS.num_collections.get(), initial_count + 1);

        // Add another collection
        catalog
            .add_collection("test2".to_string(), collection.clone())
            .await;

        // Verify metric was incremented again
        assert_eq!(INTERNAL_METRICS.num_collections.get(), initial_count + 2);
    }
}
