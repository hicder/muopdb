use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use dashmap::DashMap;
use snapshot::Snapshot;

use crate::index::Searchable;

pub mod snapshot;

pub struct Version {
    toc: Vec<String>,
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
    versions: DashMap<u64, Version>,
    all_segments: DashMap<String, Arc<dyn Searchable>>,

    versions_info: Mutex<VersionsInfo>,
}

impl Collection {
    pub fn new() -> Self {
        Self {
            versions: DashMap::new(),
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
    use std::sync::Arc;

    use anyhow::Result;

    use crate::collection::Collection;

    #[test]
    fn test_collection() -> Result<()> {
        // let dir = tempdir()?;
        let collection = Arc::new(Collection::new());

        // let segment1 = Arc::new(DashMap::new());
        // collection.add_segment("segment1", segment1.clone());

        // let segment2 = Arc::new(DashMap::new());
        // collection.add_segment("segment2", segment2.clone());

        {
            let snapshot = collection.get_snapshot()?;
            assert_eq!(snapshot.version(), 0);
        }

        // let segment3 = Arc::new(DashMap::new());
        // collection.add_segment("segment3", segment3.clone());

        Ok(())
    }
}
