pub mod reader;
pub mod snapshot;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use anyhow::{Ok, Result};
use config::collection::CollectionConfig;
use config::enums::QuantizerType;
use dashmap::DashMap;
use fs_extra::dir::CopyOptions;
use lock_api::RwLockUpgradableReadGuard;
use parking_lot::{RawRwLock, RwLock};
use quantization::noq::noq::NoQuantizer;
use quantization::pq::pq::ProductQuantizer;
use serde::{Deserialize, Serialize};
use snapshot::Snapshot;
use utils::distance::l2::L2DistanceCalculator;

use crate::index::Searchable;
use crate::multi_spann::reader::MultiSpannReader;
use crate::optimizers::SegmentOptimizer;
use crate::segment::immutable_segment::ImmutableSegment;
use crate::segment::mutable_segment::MutableSegment;
use crate::segment::pending_segment::PendingSegment;
use crate::segment::{BoxedImmutableSegment, Segment};

pub trait SegmentSearchable: Searchable + Segment {}
pub type BoxedSegmentSearchable = Box<dyn SegmentSearchable + Send + Sync>;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TableOfContent {
    pub toc: Vec<String>,

    #[serde(default)]
    pub pending: HashMap<String, Vec<String>>,
}

impl TableOfContent {
    pub fn new(toc: Vec<String>) -> Self {
        Self {
            toc,
            pending: HashMap::new(),
        }
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
    all_segments: DashMap<String, BoxedImmutableSegment>,
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

    pub fn init_new_collection(base_directory: String, config: &CollectionConfig) -> Result<()> {
        std::fs::create_dir_all(base_directory.clone())?;

        // Write version 0
        let toc_path = format!("{}/version_0", base_directory);
        let toc = TableOfContent {
            toc: vec![],
            pending: HashMap::new(),
        };
        serde_json::to_writer_pretty(std::fs::File::create(toc_path)?, &toc)?;

        // Write the config file
        let config_path = format!("{}/collection_config.json", base_directory);
        serde_json::to_writer_pretty(std::fs::File::create(config_path)?, config)?;

        Ok(())
    }

    pub fn init_from(
        base_directory: String,
        version: u64,
        toc: TableOfContent,
        segments: Vec<BoxedImmutableSegment>,
        segment_config: CollectionConfig,
    ) -> Result<Self> {
        let versions_info = RwLock::new(VersionsInfo::new());
        versions_info.write().current_version = version;
        versions_info.write().version_ref_counts.insert(version, 0);

        let all_segments = DashMap::new();
        toc.toc
            .iter()
            .zip(segments.into_iter())
            .for_each(|(name, segment)| {
                all_segments.insert(name.clone(), segment);
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

    pub fn insert(&self, doc_id: u128, data: &[f32]) -> Result<()> {
        self.mutable_segment.write().insert(doc_id, data)
    }

    pub fn insert_for_users(&self, user_ids: &[u128], doc_id: u128, data: &[f32]) -> Result<()> {
        for user_id in user_ids {
            self.mutable_segment
                .write()
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
                    let mut mutable_segment = self.mutable_segment.write();
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
                        let segment = BoxedImmutableSegment::FinalizedProductQuantizationSegment(
                            Arc::new(RwLock::new(ImmutableSegment::new(
                                index,
                                name_for_new_segment.clone(),
                            ))),
                        );
                        self.add_segments(vec![name_for_new_segment], vec![segment])
                    }
                    QuantizerType::NoQuantizer => {
                        let index = spann_reader.read::<NoQuantizer<L2DistanceCalculator>>()?;
                        let segment = BoxedImmutableSegment::FinalizedNoQuantizationSegment(
                            Arc::new(RwLock::new(ImmutableSegment::new(
                                index,
                                name_for_new_segment.clone(),
                            ))),
                        );
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
        let segments: Vec<BoxedImmutableSegment> = toc
            .iter()
            .map(|name| self.all_segments.get(name).unwrap().value().clone())
            .collect();
        Ok(Snapshot::new(
            segments,
            current_version_number,
            Arc::clone(&self),
        ))
    }

    pub fn get_current_toc(&self) -> TableOfContent {
        self.versions
            .get(&self.current_version())
            .unwrap()
            .value()
            .clone()
    }

    /// Add segments to the collection, effectively creating a new version.
    pub fn add_segments(
        &self,
        names: Vec<String>,
        segments: Vec<BoxedImmutableSegment>,
    ) -> Result<()> {
        for (name, segment) in names.iter().zip(segments) {
            self.all_segments.insert(name.clone(), segment);
        }

        // Under the upgradable read lock, we do the following:
        // - Increment the current version
        // - Add the new version to the toc, and persist to disk
        // Under the write lock, we do the following:
        // - Rename the tmp_version_{} to version_{}
        // - Insert the new version to the toc
        let toc_read = self.versions_info.upgradable_read();
        let current_version = toc_read.current_version;
        let new_version = current_version + 1;

        let mut new_toc = self.versions.get(&current_version).unwrap().toc.clone();
        new_toc.extend_from_slice(&names);
        let new_pending = self.versions.get(&current_version).unwrap().pending.clone();

        // Write the TOC to disk to a temporary file (with random name). Only rename atomically under the write lock.
        let tmp_toc_path = format!(
            "{}/tmp_version_{}",
            self.base_directory,
            rand::random::<u64>()
        );
        let mut tmp_toc_file = std::fs::File::create(tmp_toc_path.clone())?;
        let toc = TableOfContent {
            toc: new_toc,
            pending: new_pending,
        };
        serde_json::to_writer(&mut tmp_toc_file, &toc)?;

        // Once success, update the current version and ref counts.
        let mut toc_write = RwLockUpgradableReadGuard::upgrade(toc_read);
        let toc_path = format!("{}/version_{}", self.base_directory, new_version);
        std::fs::rename(tmp_toc_path, toc_path)?;

        toc_write.current_version = new_version;
        toc_write.version_ref_counts.insert(new_version, 0);

        self.versions.insert(new_version, toc);

        Ok(())
    }

    /// Replace old segments with a new segment.
    /// This function is not thread-safe. Caller needs to ensure the thread safety.
    fn replace_segment(
        &self,
        new_segment: BoxedImmutableSegment,
        old_segment_names: Vec<String>,
        is_pending: bool,
        toc_locked: RwLockUpgradableReadGuard<RawRwLock, VersionsInfo>,
    ) -> Result<()> {
        self.all_segments
            .insert(new_segment.name(), new_segment.clone());

        // Under the lock, we do the following:
        // - Increment the current version
        // - Add the new version to the toc, and persist to disk
        // - Insert the new version to the toc
        let current_version = toc_locked.current_version;
        let new_version = current_version + 1;

        let mut new_toc = self.versions.get(&current_version).unwrap().toc.clone();
        new_toc.retain(|name| !old_segment_names.contains(name));
        new_toc.push(new_segment.name());

        let mut new_pending = self.versions.get(&current_version).unwrap().pending.clone();
        if is_pending {
            new_pending.insert(new_segment.name(), old_segment_names);
        } else {
            // Cleanup the pending segment
            new_pending.retain(|name, _| new_toc.contains(name));
        }

        // Write the TOC to disk to a temporary file. Only rename atomically under the write lock.
        let tmp_toc_path = format!(
            "{}/tmp_version_{}",
            self.base_directory,
            rand::random::<u64>()
        );
        let mut tmp_toc_file = std::fs::File::create(tmp_toc_path.clone())?;
        let toc = TableOfContent {
            toc: new_toc,
            pending: new_pending,
        };
        serde_json::to_writer(&mut tmp_toc_file, &toc)?;

        let mut locked_versions_info = RwLockUpgradableReadGuard::upgrade(toc_locked);
        let toc_path = format!("{}/version_{}", self.base_directory, new_version);
        std::fs::rename(tmp_toc_path, toc_path)?;

        locked_versions_info.current_version = new_version;
        locked_versions_info
            .version_ref_counts
            .insert(new_version, 0);

        self.versions.insert(new_version, toc);
        Ok(())
    }

    /// Replace old segments with a new segment.
    /// This function is thread-safe.
    pub fn replace_segment_safe(
        &self,
        new_segment: BoxedImmutableSegment,
        old_segment_names: Vec<String>,
        is_pending: bool,
    ) -> Result<()> {
        let toc_locked = self.versions_info.upgradable_read();
        self.replace_segment(new_segment, old_segment_names, is_pending, toc_locked)?;
        Ok(())
    }

    pub fn current_version(&self) -> u64 {
        self.versions_info.read().current_version
    }

    pub fn get_ref_count(&self, version_number: u64) -> usize {
        self.versions_info
            .read()
            .version_ref_counts
            .get(&version_number)
            .unwrap_or(&0)
            .clone()
    }

    /// Release the ref count for the version once the snapshot is no longer needed.
    pub fn release_version(&self, version_number: u64) {
        let mut lock = self.versions_info.write();
        let count = *lock.version_ref_counts.get(&version_number).unwrap_or(&0);
        lock.version_ref_counts.insert(version_number, count - 1);
    }

    /// This is thread-safe, and will increment the ref count for the version.
    fn get_current_version_and_increment(&self) -> u64 {
        let mut lock = self.versions_info.write();
        let current_version = lock.current_version;

        let count = *lock.version_ref_counts.get(&current_version).unwrap_or(&0);
        lock.version_ref_counts.insert(current_version, count + 1);

        current_version
    }

    pub fn get_all_segment_names(&self) -> Vec<String> {
        self.all_segments
            .iter()
            .map(|pair| pair.key().clone())
            .collect()
    }

    pub fn init_optimizing(&self, segments: &Vec<String>) -> Result<String> {
        let random_name = format!("pending_segment_{}", rand::random::<u64>());
        let pending_segment_path = format!("{}/{}", self.base_directory, random_name);
        std::fs::create_dir_all(pending_segment_path.clone())?;
        let mut current_segments = Vec::new();
        for segment in segments {
            current_segments.push(self.all_segments.get(segment).unwrap().clone());
        }
        match self.segment_config.quantization_type {
            QuantizerType::ProductQuantizer => {
                let pending_segment = PendingSegment::<ProductQuantizer<L2DistanceCalculator>>::new(
                    current_segments.clone(),
                    pending_segment_path,
                );
                let new_boxed_segment = BoxedImmutableSegment::PendingProductQuantizationSegment(
                    Arc::new(RwLock::new(pending_segment)),
                );
                self.replace_segment_safe(new_boxed_segment, segments.clone(), true)?;
            }
            QuantizerType::NoQuantizer => {
                let pending_segment = PendingSegment::<NoQuantizer<L2DistanceCalculator>>::new(
                    current_segments.clone(),
                    pending_segment_path,
                );
                let new_boxed_segment = BoxedImmutableSegment::PendingNoQuantizationSegment(
                    Arc::new(RwLock::new(pending_segment)),
                );
                self.replace_segment_safe(new_boxed_segment, segments.clone(), true)?;
            }
        }

        Ok(random_name)
    }

    fn pending_to_finalized(
        &self,
        pending_segment: &str,
        toc_locked: RwLockUpgradableReadGuard<RawRwLock, VersionsInfo>,
    ) -> Result<()> {
        let random_name = format!("segment_{}", rand::random::<u64>());

        // Hardlink the pending segment to the new segment
        let pending_segment_path = format!("{}/{}", self.base_directory, pending_segment);
        let new_segment_path = format!("{}/{}", self.base_directory, random_name);

        // Create and copy content of the pending segment to the new segment
        std::fs::create_dir_all(new_segment_path.clone())?;

        let mut options = CopyOptions::default();
        options.content_only = true;
        fs_extra::dir::copy(
            pending_segment_path.clone(),
            new_segment_path.clone(),
            &options,
        )?;

        // Replace the pending segment with the new segment
        match self.segment_config.quantization_type {
            QuantizerType::ProductQuantizer => {
                let index = MultiSpannReader::new(new_segment_path.clone())
                    .read::<ProductQuantizer<L2DistanceCalculator>>()?;
                let new_segment =
                    BoxedImmutableSegment::FinalizedProductQuantizationSegment(Arc::new(
                        RwLock::new(ImmutableSegment::new(index, random_name.clone())),
                    ));
                self.replace_segment(
                    new_segment,
                    vec![pending_segment.to_string()],
                    false,
                    toc_locked,
                )?;
                Ok(())
            }
            QuantizerType::NoQuantizer => {
                let index = MultiSpannReader::new(new_segment_path)
                    .read::<NoQuantizer<L2DistanceCalculator>>()?;
                let new_segment = BoxedImmutableSegment::FinalizedNoQuantizationSegment(Arc::new(
                    RwLock::new(ImmutableSegment::new(index, random_name.clone())),
                ));
                self.replace_segment(
                    new_segment,
                    vec![pending_segment.to_string()],
                    false,
                    toc_locked,
                )?;
                Ok(())
            }
        }
    }

    pub fn run_optimizer(
        &self,
        optimizer: &impl SegmentOptimizer,
        pending_segment: &str,
    ) -> Result<()> {
        let segment = self.all_segments.get(pending_segment).unwrap().clone();
        match segment {
            BoxedImmutableSegment::PendingNoQuantizationSegment(pending_segment) => {
                let mut pending_segment = pending_segment.upgradable_read();
                optimizer.optimize(&pending_segment)?;
                pending_segment.build_index()?;
                pending_segment.try_with_upgraded(|pending_segment_write| {
                    pending_segment_write.apply_pending_deletions()?;
                    pending_segment_write.switch_to_internal_index();
                    Ok(())
                });
            }
            BoxedImmutableSegment::PendingProductQuantizationSegment(pending_segment) => {
                let mut pending_segment = pending_segment.upgradable_read();
                optimizer.optimize(&pending_segment)?;
                pending_segment.build_index()?;
                pending_segment.try_with_upgraded(|pending_segment_write| {
                    pending_segment_write.apply_pending_deletions()?;
                    pending_segment_write.switch_to_internal_index();
                    Ok(())
                });
            }
            _ => {}
        }

        // Make the pending segment finalized
        // Note that this function will upgrade toc lock.
        let toc_locked = self.versions_info.upgradable_read();
        self.pending_to_finalized(pending_segment, toc_locked)
    }
}

// Test
#[cfg(test)]
mod tests {

    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;

    use anyhow::{Ok, Result};
    use config::collection::CollectionConfig;
    use parking_lot::RwLock;
    use rand::Rng;
    use tempdir::TempDir;

    use crate::collection::Collection;
    use crate::optimizers::noop::NoopOptimizer;
    use crate::segment::{BoxedImmutableSegment, MockedSegment, Segment};
    use crate::utils::SearchContext;

    #[test]
    fn test_collection() -> Result<()> {
        let temp_dir = TempDir::new("test_collection")?;
        let base_directory: String = temp_dir.path().to_str().unwrap().to_string();
        let segment_config = CollectionConfig::default_test_config();
        let collection = Arc::new(Collection::new(base_directory.clone(), segment_config).unwrap());

        {
            let segment1: BoxedImmutableSegment =
                BoxedImmutableSegment::MockedNoQuantizationSegment(Arc::new(RwLock::new(
                    MockedSegment::new("segment1".to_string()),
                )));
            let segment2: BoxedImmutableSegment =
                BoxedImmutableSegment::MockedNoQuantizationSegment(Arc::new(RwLock::new(
                    MockedSegment::new("segment2".to_string()),
                )));

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
                        BoxedImmutableSegment::MockedNoQuantizationSegment(Arc::new(RwLock::new(
                            MockedSegment::new("segment3".to_string()),
                        ))),
                        BoxedImmutableSegment::MockedNoQuantizationSegment(Arc::new(RwLock::new(
                            MockedSegment::new("segment4".to_string()),
                        ))),
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
            let segment1 = BoxedImmutableSegment::MockedNoQuantizationSegment(Arc::new(
                RwLock::new(MockedSegment::new("segment1".to_string())),
            ));
            let segment2 = BoxedImmutableSegment::MockedNoQuantizationSegment(Arc::new(
                RwLock::new(MockedSegment::new("segment2".to_string())),
            ));

            collection_cpy
                .add_segments(
                    vec!["segment1".to_string(), "segment2".to_string()],
                    vec![segment1.clone(), segment2.clone()],
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

    #[test]
    fn test_collection_optimizer() -> Result<()> {
        let temp_dir = TempDir::new("test_collection")?;
        let base_directory: String = temp_dir.path().to_str().unwrap().to_string();
        let segment_config = CollectionConfig::default_test_config();
        let collection = Arc::new(Collection::new(base_directory.clone(), segment_config).unwrap());

        // Add a document and flush
        collection.insert_for_users(&[0], 1, &[1.0, 2.0, 3.0, 4.0])?;
        collection.flush()?;

        let segment_names = collection.get_all_segment_names();
        assert_eq!(segment_names.len(), 1);

        let pending_segment = collection.init_optimizing(&segment_names)?;

        let snapshot = collection.clone().get_snapshot()?;
        assert_eq!(snapshot.segments.len(), 1);
        assert_eq!(snapshot.version(), 2);
        let segment_name = snapshot.segments[0].name();
        assert_eq!(segment_name, pending_segment);

        let mut context = SearchContext::new(false);
        let result = snapshot
            .search_for_ids(&[0], &[1.0, 2.0, 3.0, 4.0], 10, 10, &mut context)
            .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].id, 1);

        let optimizer = NoopOptimizer::new();
        collection.run_optimizer(&optimizer, &pending_segment)?;

        let snapshot = collection.clone().get_snapshot()?;
        assert_eq!(snapshot.segments.len(), 1);
        assert_eq!(snapshot.version(), 3);

        let toc = collection.get_current_toc();
        assert_eq!(toc.toc.len(), 1);
        assert_eq!(toc.pending.len(), 0);

        Ok(())
    }

    #[test]
    fn test_collection_multi_thread_optimizer() -> Result<()> {
        let temp_dir = TempDir::new("test_collection")?;
        let base_directory: String = temp_dir.path().to_str().unwrap().to_string();
        let segment_config = CollectionConfig::default_test_config();
        let collection = Arc::new(Collection::new(base_directory.clone(), segment_config).unwrap());

        collection.insert_for_users(&[0], 1, &[1.0, 2.0, 3.0, 4.0])?;
        collection.flush()?;

        // A thread to optimize the segment
        let collection_cpy_for_optimizer = collection.clone();
        let collection_cpy_for_query = collection.clone();

        let stopped = Arc::new(AtomicBool::new(false));
        let stopped_cpy_for_optimizer = stopped.clone();
        std::thread::spawn(move || {
            while !stopped_cpy_for_optimizer.load(std::sync::atomic::Ordering::Relaxed) {
                let c = collection_cpy_for_optimizer.clone();
                let snapshot = c.get_snapshot().unwrap();
                let segment_names = snapshot.segments.iter().map(|s| s.name()).collect();
                let pending_segment = collection_cpy_for_optimizer
                    .init_optimizing(&segment_names)
                    .unwrap();

                let toc = collection_cpy_for_optimizer.get_current_toc();
                assert_eq!(toc.pending.len(), 1);

                // Sleep randomly between 100ms and 200ms
                let sleep_duration = rand::thread_rng().gen_range(100..200);
                std::thread::sleep(std::time::Duration::from_millis(sleep_duration));

                let optimizer = NoopOptimizer::new();
                collection_cpy_for_optimizer
                    .run_optimizer(&optimizer, &pending_segment)
                    .unwrap();

                let toc = collection_cpy_for_optimizer.get_current_toc();
                assert_eq!(toc.pending.len(), 0);
            }
        });

        // A thread to query the collection
        let stopped_cpy_for_query = stopped.clone();
        std::thread::spawn(move || {
            while !stopped_cpy_for_query.load(std::sync::atomic::Ordering::Relaxed) {
                let c = collection_cpy_for_query.clone();
                let snapshot = c.get_snapshot().unwrap();
                let mut context = SearchContext::new(false);
                let result = snapshot
                    .search_for_ids(&[0], &[1.0, 2.0, 3.0, 4.0], 10, 10, &mut context)
                    .unwrap();
                assert_eq!(result.len(), 1);
                assert_eq!(result[0].id, 1);

                // Sleep randomly between 100ms and 200ms
                let sleep_duration = rand::thread_rng().gen_range(100..200);
                std::thread::sleep(std::time::Duration::from_millis(sleep_duration));
            }
        });

        // Sleep for 5 seconds, then stop the threads
        std::thread::sleep(std::time::Duration::from_millis(5000));
        stopped.store(true, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }
}
