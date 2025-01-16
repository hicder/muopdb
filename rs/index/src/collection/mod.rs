pub mod reader;
pub mod snapshot;

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

use anyhow::{Ok, Result};
use config::collection::CollectionConfig;
use config::enums::QuantizerType;
use dashmap::DashMap;
use quantization::noq::noq::NoQuantizer;
use quantization::pq::pq::ProductQuantizer;
use serde::{Deserialize, Serialize};
use snapshot::Snapshot;
use utils::distance::l2::L2DistanceCalculator;

use crate::index::Searchable;
use crate::multi_spann::reader::MultiSpannReader;
use crate::segment::immutable_segment::ImmutableSegment;
use crate::segment::mutable_segment::MutableSegment;
use crate::segment::Segment;

pub trait SegmentSearchable: Searchable + Segment {}
pub type BoxedSegmentSearchable = Box<dyn SegmentSearchable + Send + Sync>;

#[derive(Serialize, Deserialize, Debug)]
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
    pub versions: DashMap<u64, TableOfContent>,
    all_segments: DashMap<String, Arc<BoxedSegmentSearchable>>,
    versions_info: RwLock<VersionsInfo>,
    base_directory: String,
    mutable_segment: RwLock<MutableSegment>,
    segment_config: CollectionConfig,

    // A mutex for flushing
    flushing: Mutex<()>,
}

impl Collection {
    pub fn new(base_directory: String, segment_config: CollectionConfig) -> Result<Self> {
        let versions: DashMap<u64, TableOfContent> = DashMap::new();
        versions.insert(0, TableOfContent::new(vec![]));

        // Create a new segment_config with a random name
        let random_name = format!("tmp_segment_{}", rand::random::<u64>());
        let segment_base_directory = format!("{}/{}", base_directory, random_name);

        let mutable_segment = RwLock::new(MutableSegment::new(
            segment_config.clone(),
            segment_base_directory,
        )?);

        Ok(Self {
            versions,
            all_segments: DashMap::new(),
            versions_info: RwLock::new(VersionsInfo::new()),
            base_directory,
            mutable_segment,
            segment_config,
            flushing: Mutex::new(()),
        })
    }

    pub fn init_new_collection(
        base_directory: String,
        config: &CollectionConfig,
    ) -> Result<()> {
        std::fs::create_dir_all(base_directory.clone())?;

        // Write version 0
        let toc_path = format!("{}/version_0", base_directory);
        let toc = TableOfContent { toc: vec![] };
        serde_json::to_writer(std::fs::File::create(toc_path)?, &toc)?;

        // Write the config file
        let config_path = format!("{}/collection_config.json", base_directory);
        serde_json::to_writer(std::fs::File::create(config_path)?, config)?;

        Ok(())
    }

    pub fn init_from(
        base_directory: String,
        version: u64,
        toc: TableOfContent,
        segments: Vec<Arc<BoxedSegmentSearchable>>,
        segment_config: CollectionConfig,
    ) -> Result<Self> {
        let versions_info = RwLock::new(VersionsInfo::new());
        versions_info.write().unwrap().current_version = version;
        versions_info
            .write()
            .unwrap()
            .version_ref_counts
            .insert(version, 0);

        let all_segments = DashMap::new();
        toc.toc
            .iter()
            .zip(segments.iter())
            .for_each(|(name, segment)| {
                all_segments.insert(name.clone(), segment.clone());
            });

        let versions = DashMap::new();
        versions.insert(version, toc);

        // Create a new segment_config with a random name
        let random_name = format!("tmp_segment_{}", rand::random::<u64>());
        let random_base_directory = format!("{}/{}", base_directory, random_name);
        let mutable_segment = RwLock::new(MutableSegment::new(
            segment_config.clone(),
            random_base_directory,
        )?);

        Ok(Self {
            versions,
            all_segments,
            versions_info,
            base_directory,
            mutable_segment,
            segment_config,
            flushing: Mutex::new(()),
        })
    }

    pub fn insert(&self, doc_id: u64, data: &[f32]) -> Result<()> {
        self.mutable_segment.write().unwrap().insert(doc_id, data)
    }

    pub fn insert_for_users(&self, user_ids: &[u64], doc_id: u64, data: &[f32]) -> Result<()> {
        for user_id in user_ids {
            self.mutable_segment
                .write()
                .unwrap()
                .insert_for_user(*user_id, doc_id, data)?;
        }
        Ok(())
    }

    pub fn dimensions(&self) -> usize {
        self.segment_config.num_features
    }

    /// Turns mutable segment into immutable one, which is the only queryable segment type
    /// currently.
    pub fn flush(&self) -> Result<()> {
        // Try to acquire the flushing lock. If it fails, then another thread is already flushing.
        // This is a best effort approach, and we don't want to block the main thread.
        match self.flushing.try_lock() {
            std::result::Result::Ok(_) => {
                let tmp_name = format!("tmp_segment_{}", rand::random::<u64>());
                let writable_base_directory = format!("{}/{}", self.base_directory, tmp_name);
                let mut new_writable_segment =
                    MutableSegment::new(self.segment_config.clone(), writable_base_directory)?;

                {
                    // Grab the write lock and swap tmp_segment with mutable_segment
                    let mut mutable_segment = self.mutable_segment.write().unwrap();
                    std::mem::swap(&mut *mutable_segment, &mut new_writable_segment);
                }

                let name_for_new_segment = format!("segment_{}", rand::random::<u64>());
                new_writable_segment
                    .build(self.base_directory.clone(), name_for_new_segment.clone())?;

                // Read the segment
                let spann_reader = MultiSpannReader::new(format!(
                    "{}/{}",
                    self.base_directory, name_for_new_segment
                ));
                match self.segment_config.quantization_type {
                    QuantizerType::ProductQuantizer => {
                        let index =
                            spann_reader.read::<ProductQuantizer<L2DistanceCalculator>>()?;
                        let segment: Arc<Box<dyn SegmentSearchable + Send + Sync>> =
                            Arc::new(Box::new(ImmutableSegment::new(index)));

                        self.add_segments(vec![name_for_new_segment], vec![segment])
                    }
                    QuantizerType::NoQuantizer => {
                        let index = spann_reader.read::<NoQuantizer<L2DistanceCalculator>>()?;
                        let segment: Arc<Box<dyn SegmentSearchable + Send + Sync>> =
                            Arc::new(Box::new(ImmutableSegment::new(index)));

                        self.add_segments(vec![name_for_new_segment], vec![segment])
                    }
                }
            }
            Err(_) => {
                return Err(anyhow::anyhow!("Another thread is already flushing"));
            }
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
    pub fn add_segments(
        &self,
        names: Vec<String>,
        segments: Vec<Arc<BoxedSegmentSearchable>>,
    ) -> Result<()> {
        for (name, segment) in names.iter().zip(segments) {
            self.all_segments.insert(name.clone(), segment);
        }

        // Under the lock, we do the following:
        // - Increment the current version
        // - Add the new version to the toc, and persist to disk
        // - Insert the new version to the toc
        let mut locked_versions_info = self.versions_info.write().unwrap();
        let current_version = locked_versions_info.current_version;
        let new_version = current_version + 1;

        let mut new_toc = self.versions.get(&current_version).unwrap().toc.clone();
        new_toc.extend_from_slice(&names);

        // Write the TOC to disk.
        let toc_path = format!("{}/version_{}", self.base_directory, new_version);
        let toc = TableOfContent { toc: new_toc };
        serde_json::to_writer(std::fs::File::create(toc_path)?, &toc)?;

        // Once success, update the current version and ref counts.
        locked_versions_info.current_version = new_version;
        locked_versions_info
            .version_ref_counts
            .insert(new_version, 0);

        self.versions.insert(new_version, toc);

        Ok(())
    }

    pub fn current_version(&self) -> u64 {
        self.versions_info.read().unwrap().current_version
    }

    pub fn get_ref_count(&self, version_number: u64) -> usize {
        self.versions_info
            .read()
            .unwrap()
            .version_ref_counts
            .get(&version_number)
            .unwrap_or(&0)
            .clone()
    }

    /// Release the ref count for the version once the snapshot is no longer needed.
    pub fn release_version(&self, version_number: u64) {
        let mut lock = self.versions_info.write().unwrap();
        let count = *lock.version_ref_counts.get(&version_number).unwrap_or(&0);
        lock.version_ref_counts.insert(version_number, count - 1);
    }

    /// This is thread-safe, and will increment the ref count for the version.
    fn get_current_version_and_increment(&self) -> u64 {
        let mut lock = self.versions_info.write().unwrap();
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
    use config::collection::CollectionConfig;
    use tempdir::TempDir;

    use super::SegmentSearchable;
    use crate::collection::{BoxedSegmentSearchable, Collection};
    use crate::index::Searchable;
    use crate::segment::Segment;

    struct MockSearchable {}

    impl MockSearchable {
        pub fn new() -> Self {
            Self {}
        }
    }

    impl SegmentSearchable for MockSearchable {}

    impl Segment for MockSearchable {
        fn insert(&mut self, _doc_id: u64, _data: &[f32]) -> Result<()> {
            todo!()
        }

        fn remove(&mut self, _doc_id: u64) -> Result<bool> {
            todo!()
        }

        fn may_contains(&self, _doc_id: u64) -> bool {
            todo!()
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
            todo!()
        }
    }

    #[test]
    fn test_collection() -> Result<()> {
        let temp_dir = TempDir::new("test_collection")?;
        let base_directory: String = temp_dir.path().to_str().unwrap().to_string();
        let segment_config = CollectionConfig::default_test_config();
        let collection = Arc::new(Collection::new(base_directory.clone(), segment_config).unwrap());

        {
            let segment1: Arc<BoxedSegmentSearchable> = Arc::new(Box::new(MockSearchable::new()));
            let segment2: Arc<BoxedSegmentSearchable> = Arc::new(Box::new(MockSearchable::new()));

            collection
                .add_segments(
                    vec!["segment1".to_string(), "segment2".to_string()],
                    vec![segment1.clone(), segment2.clone()],
                )
                .unwrap();
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

            collection
                .add_segments(
                    vec!["segment3".to_string(), "segment4".to_string()],
                    vec![
                        Arc::new(Box::new(MockSearchable::new())),
                        Arc::new(Box::new(MockSearchable::new())),
                    ],
                )
                .unwrap();

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
        let temp_dir = TempDir::new("test_collection")?;
        let base_directory: String = temp_dir.path().to_str().unwrap().to_string();
        let segment_config = CollectionConfig::default_test_config();

        let collection = Arc::new(Collection::new(base_directory.clone(), segment_config).unwrap());
        let stopped = Arc::new(AtomicBool::new(false));

        // Create a thread to add segments, and let it runs for a while
        let stopped_cpy = stopped.clone();
        let collection_cpy = collection.clone();
        std::thread::spawn(move || {
            let segment1: Arc<BoxedSegmentSearchable> = Arc::new(Box::new(MockSearchable::new()));
            let segment2: Arc<BoxedSegmentSearchable> = Arc::new(Box::new(MockSearchable::new()));

            collection_cpy
                .add_segments(
                    vec!["segment1".to_string(), "segment2".to_string()],
                    vec![segment1, segment2],
                )
                .unwrap();

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
