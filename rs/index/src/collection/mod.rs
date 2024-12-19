use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use dashmap::DashMap;
use snapshot::Snapshot;

use crate::index::BoxedSearchable;

pub mod snapshot;

pub struct TableOfContent {
    pub toc: Vec<String>,
}

impl TableOfContent {
    pub fn new(toc: Vec<String>) -> Self {
        Self { toc }
    }
}

pub struct VersionsInfo {
    pub current_version: u64,
    pub version_ref_counts: HashMap<u64, usize>,
}

impl VersionsInfo {
    pub fn new() -> Self {
        let mut version_ref_counts = HashMap::new();
        version_ref_counts.insert(0, 0);
        Self {
            current_version: 0,
            version_ref_counts,
        }
    }
}

/// Collection is thread-safe. All pub fn are thread-safe.
/// TODO(hicder): Add open segment to add documents.
pub struct Collection {
    versions: DashMap<u64, TableOfContent>,
    all_segments: DashMap<String, Arc<BoxedSearchable>>,

    versions_info: Mutex<VersionsInfo>,
}

impl Collection {
    pub fn new() -> Self {
        let versions: DashMap<u64, TableOfContent> = DashMap::new();
        versions.insert(0, TableOfContent::new(vec![]));

        Self {
            versions,
            all_segments: DashMap::new(),
            versions_info: Mutex::new(VersionsInfo::new()),
        }
    }

    /// Get a consistent snapshot for the collection
    /// TODO(hicder): Get the consistent snapshot w.r.t. time.
    pub fn get_snapshot(self: Arc<Self>) -> Result<Snapshot> {
        if self.versions.is_empty() {
            return Err(anyhow::anyhow!("Collection is empty"));
        }

        let current_version_number = self.get_current_version_and_increment();
        let latest_version = self.versions.get(&current_version_number);
        if latest_version.is_none() {
            // It shouldn't happen, but just in case, we still release the version
            self.release_version(current_version_number);
            return Err(anyhow::anyhow!("Collection is empty"));
        }

        let toc = latest_version.unwrap().toc.clone();
        Ok(Snapshot::new(
            toc.iter()
                .map(|name| self.all_segments.get(name).unwrap().clone())
                .collect(),
            current_version_number,
            Arc::clone(&self),
        ))
    }

    /// Add segments to the collection, effectively creating a new version.
    pub fn add_segments(&self, names: Vec<String>, segments: Vec<Arc<BoxedSearchable>>) {
        let mut locked_versions_info = self.versions_info.lock().unwrap();
        let current_version = locked_versions_info.current_version;
        let new_version = current_version + 1;

        let mut new_toc = self.versions.get(&current_version).unwrap().toc.clone();
        new_toc.extend_from_slice(&names);

        locked_versions_info.current_version = new_version;
        locked_versions_info
            .version_ref_counts
            .insert(new_version, 0);

        self.versions
            .insert(new_version, TableOfContent { toc: new_toc });
        for (name, segment) in names.iter().zip(segments) {
            self.all_segments.insert(name.clone(), segment);
        }
    }

    pub fn current_version(&self) -> u64 {
        self.versions_info.lock().unwrap().current_version
    }

    pub fn get_ref_count(&self, version_number: u64) -> usize {
        self.versions_info
            .lock()
            .unwrap()
            .version_ref_counts
            .get(&version_number)
            .unwrap_or(&0)
            .clone()
    }

    /// Release the ref count for the version once the snapshot is no longer needed.
    pub fn release_version(&self, version_number: u64) {
        let mut lock = self.versions_info.lock().unwrap();
        let count = *lock.version_ref_counts.get(&version_number).unwrap_or(&0);
        lock.version_ref_counts.insert(version_number, count - 1);
    }

    /// This is thread-safe, and will increment the ref count for the version.
    fn get_current_version_and_increment(&self) -> u64 {
        let mut lock = self.versions_info.lock().unwrap();
        let current_version = lock.current_version;

        let count = *lock.version_ref_counts.get(&current_version).unwrap_or(&0);
        lock.version_ref_counts.insert(current_version, count + 1);

        current_version
    }
}

// Test
#[cfg(test)]
mod tests {

    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;

    use anyhow::{Ok, Result};

    use crate::collection::Collection;
    use crate::index::{BoxedSearchable, Searchable};

    struct MockSearchable {}

    impl MockSearchable {
        pub fn new() -> Self {
            Self {}
        }
    }

    impl Searchable for MockSearchable {
        fn search(
            &self,
            _query: &[f32],
            _k: usize,
            _ef_construction: u32,
            _context: &mut crate::utils::SearchContext,
        ) -> Option<Vec<crate::utils::IdWithScore>> {
            // Don't do anything. Just for testing purpose.
            None
        }
    }

    #[test]
    fn test_collection() -> Result<()> {
        // let dir = tempdir()?;
        let collection = Arc::new(Collection::new());

        {
            let segment1: Arc<BoxedSearchable> = Arc::new(Box::new(MockSearchable::new()));
            let segment2: Arc<BoxedSearchable> = Arc::new(Box::new(MockSearchable::new()));

            collection.add_segments(
                vec!["segment1".to_string(), "segment2".to_string()],
                vec![segment1.clone(), segment2.clone()],
            );
        }
        let current_version = collection.current_version();
        assert_eq!(current_version, 1);

        let version_1 = 1;
        {
            let snapshot = collection.clone().get_snapshot()?;
            assert_eq!(snapshot.segments.len(), 2);

            let ref_count = collection.clone().get_ref_count(version_1);
            assert_eq!(ref_count, 1);
        }

        // Snapshot should be dropped when it goes out of scope
        let ref_count = collection.clone().get_ref_count(version_1);
        assert_eq!(ref_count, 0);

        // Create another snapshot, then add new segments
        let version_2 = 2;
        {
            let snapshot = collection.clone().get_snapshot()?;
            assert_eq!(snapshot.segments.len(), 2);
            assert_eq!(snapshot.version(), 1);

            collection.add_segments(
                vec!["segment3".to_string(), "segment4".to_string()],
                vec![
                    Arc::new(Box::new(MockSearchable::new())),
                    Arc::new(Box::new(MockSearchable::new())),
                ],
            );

            let ref_count = collection.clone().get_ref_count(version_1);
            assert_eq!(ref_count, 1);

            let ref_count = collection.clone().get_ref_count(version_2);
            assert_eq!(ref_count, 0);
        }

        // Snapshot should be dropped when it goes out of scope
        let ref_count = collection.clone().get_ref_count(version_1);
        assert_eq!(ref_count, 0);
        let ref_count = collection.clone().get_ref_count(version_2);
        assert_eq!(ref_count, 0);

        Ok(())
    }

    #[test]
    fn test_collection_multi_thread() -> Result<()> {
        let collection = Arc::new(Collection::new());
        let stopped = Arc::new(AtomicBool::new(false));

        // Create a thread to add segments, and let it runs for a while
        let stopped_cpy = stopped.clone();
        let collection_cpy = collection.clone();
        std::thread::spawn(move || {
            let segment1: Arc<BoxedSearchable> = Arc::new(Box::new(MockSearchable::new()));
            let segment2: Arc<BoxedSearchable> = Arc::new(Box::new(MockSearchable::new()));

            collection_cpy.add_segments(
                vec!["segment1".to_string(), "segment2".to_string()],
                vec![segment1, segment2],
            );

            while !stopped_cpy.load(std::sync::atomic::Ordering::Relaxed) {
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
        });

        // Sleep until there is a new version
        let mut latest_version = collection.clone().current_version();
        while latest_version != 1 {
            std::thread::sleep(std::time::Duration::from_millis(100));
            latest_version = collection.clone().current_version();
        }

        // Create another thread to get a snapshot
        let collection_cpy = collection.clone();
        let stopped_cpy = stopped.clone();
        std::thread::spawn(move || {
            let snapshot = collection_cpy.clone().get_snapshot().unwrap();
            assert_eq!(snapshot.segments.len(), 2);
            assert_eq!(snapshot.version(), 1);

            let version_1 = 1;
            let ref_count = collection_cpy.clone().get_ref_count(version_1);
            assert_eq!(ref_count, 1);

            while !stopped_cpy.load(std::sync::atomic::Ordering::Relaxed) {
                assert_eq!(snapshot.version(), 1);
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
        });

        // Sleep for 200ms, then check ref count
        std::thread::sleep(std::time::Duration::from_millis(500));
        let version_1 = 1;
        let ref_count = collection.clone().get_ref_count(version_1);
        assert_eq!(ref_count, 1);

        std::thread::sleep(std::time::Duration::from_millis(500));
        stopped.store(true, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }
}
