use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{Ok, Result};
use atomic_refcell::AtomicRefCell;
use config::collection::CollectionConfig;
use dashmap::DashMap;
use fs_extra::dir::CopyOptions;
use lock_api::RwLockUpgradableReadGuard;
use log::{debug, info, warn};
use metrics::INTERNAL_METRICS;
use parking_lot::{RawRwLock, RwLock};
use quantization::quantization::Quantizer;
use tokio::sync::mpsc::{self, Receiver, Sender};

use super::snapshot::Snapshot;
use super::{OpChannelEntry, TableOfContent, VersionsInfo};
use crate::multi_spann::reader::MultiSpannReader;
use crate::optimizers::merge::MergeOptimizer;
use crate::optimizers::vacuum::VacuumOptimizer;
use crate::optimizers::SegmentOptimizer;
use crate::segment::immutable_segment::ImmutableSegment;
use crate::segment::mutable_segment::MutableSegment;
use crate::segment::pending_mutable_segment::PendingMutableSegment;
use crate::segment::pending_segment::PendingSegment;
use crate::segment::{BoxedImmutableSegment, Segment};
use crate::wal::entry::WalOpType;
use crate::wal::wal::Wal;

pub struct SegmentInfo {
    pub name: String,
    pub size_in_bytes: u64,
    pub num_docs: u64,
}

pub struct SegmentInfoAndVersion {
    pub segment_infos: Vec<SegmentInfo>,
    pub version: u64,
}

/// MutableSegments is protected by a RwLock. This is to ensure atomicity when swapping
/// the inner mutable segments during flush.
/// On insert, grab the read-lock of this struct, then read-lock of mutable_segment.
/// On invalidate, grab read-lock of this struct, then read-lock of both.
/// On flush, grab write-lock of this struct, then write-lock of both.
struct MutableSegments {
    pub mutable_segment: RwLock<MutableSegment>,
    pub pending_mutable_segment: RwLock<Option<PendingMutableSegment>>,
}

/// Collection is thread-safe. All pub fn are thread-safe.
pub struct Collection<Q: Quantizer + Clone + Send + Sync> {
    pub versions: DashMap<u64, TableOfContent>,
    all_segments: DashMap<String, BoxedImmutableSegment<Q>>,
    versions_info: RwLock<VersionsInfo>,
    base_directory: String,
    collection_name: String,
    mutable_segments: RwLock<MutableSegments>,
    segment_config: CollectionConfig,
    wal: Option<RwLock<Wal>>,

    // A channel for sending ops to collection
    // Channels are thread-safe, so no lock is required when processing ops
    sender: Sender<OpChannelEntry>,
    receiver: AtomicRefCell<Receiver<OpChannelEntry>>,

    last_flush_time: Mutex<Instant>,

    // A mutex for flushing
    flushing: Mutex<()>,
}

impl<Q: Quantizer + Clone + Send + Sync + 'static> Collection<Q> {
    pub fn new(
        collection_name: String,
        base_directory: String,
        segment_config: CollectionConfig,
    ) -> Result<Self> {
        let versions: DashMap<u64, TableOfContent> = DashMap::new();
        versions.insert(0, TableOfContent::new(vec![]));

        // Create a new segment_config with a random name
        let random_name = format!("tmp_segment_{}", rand::random::<u64>());
        let segment_base_directory = format!("{}/{}", base_directory, random_name);

        let mutable_segment = RwLock::new(MutableSegment::new(
            segment_config.clone(),
            segment_base_directory,
        )?);

        let wal = if segment_config.wal_file_size > 0 {
            let wal_directory = format!("{}/wal", base_directory);
            Some(RwLock::new(Wal::open(
                &wal_directory,
                segment_config.wal_file_size,
                -1,
            )?))
        } else {
            None
        };

        let (sender, receiver) = mpsc::channel(100);
        let receiver = AtomicRefCell::new(receiver);

        // Initialize the metrics
        INTERNAL_METRICS.num_active_segments_set(&collection_name, 0);
        INTERNAL_METRICS.num_searchable_docs_set(&collection_name, 0);

        Ok(Self {
            versions,
            all_segments: DashMap::new(),
            versions_info: RwLock::new(VersionsInfo::new()),
            base_directory,
            collection_name,
            mutable_segments: RwLock::new(MutableSegments {
                mutable_segment,
                pending_mutable_segment: RwLock::new(None),
            }),
            segment_config,
            flushing: Mutex::new(()),
            wal,
            sender,
            receiver,
            last_flush_time: Mutex::new(Instant::now()),
        })
    }

    pub fn init_new_collection(base_directory: String, config: &CollectionConfig) -> Result<()> {
        std::fs::create_dir_all(base_directory.clone())?;

        // Write version 0
        let toc_path = format!("{}/version_0", base_directory);
        let toc = TableOfContent::default();
        serde_json::to_writer_pretty(std::fs::File::create(toc_path)?, &toc)?;

        // Write the config file
        let config_path = format!("{}/collection_config.json", base_directory);
        serde_json::to_writer_pretty(std::fs::File::create(config_path)?, config)?;

        Ok(())
    }

    pub fn init_from(
        collection_name: String,
        base_directory: String,
        version: u64,
        toc: TableOfContent,
        segments: Vec<BoxedImmutableSegment<Q>>,
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
        let last_sequence_number = toc.sequence_number;

        let versions = DashMap::new();
        versions.insert(version, toc);

        // Remove all directories whose name starts with tmp_segment_
        let base_dir = std::path::Path::new(&base_directory);
        if base_dir.exists() {
            for entry in std::fs::read_dir(base_dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.is_dir() {
                    if let Some(file_name) = path.file_name() {
                        if let Some(file_name_str) = file_name.to_str() {
                            if file_name_str.starts_with("tmp_segment_") {
                                std::fs::remove_dir_all(path)?;
                            }
                        }
                    }
                }
            }
        }

        // Create a new segment_config with a random name
        let random_name = format!("tmp_segment_{}", rand::random::<u64>());
        let random_base_directory = format!("{}/{}", base_directory, random_name);
        std::fs::create_dir_all(&random_base_directory)?;
        let mutable_segment = MutableSegment::new(segment_config.clone(), random_base_directory)?;

        let wal = if segment_config.wal_file_size > 0 {
            let wal_directory = format!("{}/wal", base_directory);
            let wal = Wal::open(
                &wal_directory,
                segment_config.wal_file_size,
                last_sequence_number as i64,
            )?;
            let iterators = wal.get_iterators();
            let mut seq_no = (last_sequence_number + 1) as i64;
            for mut iterator in iterators {
                if iterator.last_seq_no() < seq_no {
                    debug!(
                        "Skipping iterator with last seq_no: {}",
                        iterator.last_seq_no()
                    );
                    continue;
                }

                iterator.skip_to(seq_no)?;
                for op in iterator {
                    if let anyhow::Result::Ok(wal_entry) = op {
                        let entry_seq_no = wal_entry.seq_no;
                        debug!("Processing op with seq_no: {}", entry_seq_no);
                        let wal_entry = wal_entry.decode(segment_config.num_features);
                        match wal_entry.op_type {
                            WalOpType::Insert => {
                                let doc_ids = wal_entry.doc_ids;
                                let user_ids = wal_entry.user_ids;
                                let data = wal_entry.data;

                                let mut num_successful_inserts: i64 = 0;
                                data.chunks(segment_config.num_features)
                                    .zip(doc_ids)
                                    .for_each(|(vector, doc_id)| {
                                        for user_id in user_ids {
                                            mutable_segment
                                                .insert_for_user(
                                                    *user_id,
                                                    *doc_id,
                                                    vector,
                                                    entry_seq_no,
                                                )
                                                .unwrap();

                                            num_successful_inserts += 1;
                                        }
                                    });

                                // The doc is not immediately searchable, but we update the metrics anyway
                                INTERNAL_METRICS.num_searchable_docs_inc_by(
                                    &collection_name,
                                    num_successful_inserts,
                                );
                            }
                            WalOpType::Delete => {
                                let doc_ids = wal_entry.doc_ids;
                                let user_ids = wal_entry.user_ids;
                                assert!(wal_entry.data.is_empty());
                                let ver = versions.get(&version).unwrap();

                                let mut num_successful_deletes: i64 = 0;
                                user_ids.iter().try_for_each(|&user_id| {
                                    doc_ids.iter().try_for_each(|&doc_id| {
                                        Self::remove_impl(
                                            &ver,
                                            &all_segments,
                                            &mutable_segment,
                                            /* pending_mutable_segment */ None,
                                            user_id,
                                            doc_id,
                                            entry_seq_no,
                                        )?;

                                        num_successful_deletes += 1;
                                        Ok(())
                                    })
                                })?;

                                // The docs are immediately not searchable
                                INTERNAL_METRICS.num_searchable_docs_dec_by(
                                    &collection_name,
                                    num_successful_deletes,
                                );
                            }
                        }
                    } else {
                        break;
                    }
                    seq_no += 1;
                }
            }

            Some(RwLock::new(wal))
        } else {
            None
        };

        let (sender, receiver) = mpsc::channel(100);
        let receiver = AtomicRefCell::new(receiver);

        INTERNAL_METRICS.num_active_segments_set(&collection_name, all_segments.len() as i64);

        Ok(Self {
            versions,
            all_segments,
            versions_info,
            base_directory,
            collection_name,
            mutable_segments: RwLock::new(MutableSegments {
                mutable_segment: RwLock::new(mutable_segment),
                pending_mutable_segment: RwLock::new(None),
            }),
            segment_config,
            flushing: Mutex::new(()),
            wal,
            sender,
            receiver,
            last_flush_time: Mutex::new(Instant::now()),
        })
    }

    pub fn use_wal(&self) -> bool {
        self.wal.is_some()
    }

    pub fn should_auto_flush(&self) -> bool {
        if self.segment_config.max_pending_ops == 0 && self.segment_config.max_time_to_flush_ms == 0
        {
            return false;
        }

        if self.segment_config.max_pending_ops > 0 {
            let current_seq_no = self
                .mutable_segments
                .read()
                .mutable_segment
                .read()
                .last_sequence_number();
            let flushed_seq_no = self
                .versions
                .get(&self.current_version())
                .unwrap()
                .sequence_number;
            if current_seq_no as i64 - flushed_seq_no >= self.segment_config.max_pending_ops as i64
            {
                debug!(
                    "Flushing because of max pending ops: {}",
                    current_seq_no as i64 - flushed_seq_no as i64
                );
                return true;
            }
        }

        if self.segment_config.max_time_to_flush_ms > 0 {
            let last_flush_time = self.last_flush_time.lock().unwrap().clone();
            let current_time = std::time::Instant::now();
            if current_time.duration_since(last_flush_time)
                >= std::time::Duration::from_millis(self.segment_config.max_time_to_flush_ms)
            {
                debug!(
                    "Flushing because of max time to flush: {:?}",
                    current_time.duration_since(last_flush_time)
                );
                return true;
            }
        }

        false
    }

    pub async fn write_to_wal(
        &self,
        doc_ids: &[u128],
        user_ids: &[u128],
        data: &[f32],
        wal_op_type: WalOpType,
    ) -> Result<u64> {
        if let Some(wal) = &self.wal {
            // Write to WAL, and persist to disk.
            // Intentionally keep the write lock until we send the message to the channel.
            // This ensures that message in the channel is the same order as WAL.
            let mut wal_write = wal.write();
            let seq_no = wal_write.append(doc_ids, user_ids, data, wal_op_type.clone())?;

            // Once the WAL is written, send the op to the channel
            let op_channel_entry =
                OpChannelEntry::new(doc_ids, user_ids, data, seq_no, wal_op_type);
            self.sender.send(op_channel_entry).await?;
            Ok(seq_no)
        } else {
            Err(anyhow::anyhow!("WAL is not enabled"))
        }
    }

    pub async fn process_one_op(&self) -> Result<usize> {
        if let std::result::Result::Ok(op) = self.receiver.borrow_mut().try_recv() {
            match op.op_type {
                WalOpType::Insert => {
                    debug!("Processing insert operation with seq_no {}", op.seq_no);
                    let doc_ids = op.doc_ids();
                    let user_ids = op.user_ids();
                    let data = op.data();
                    data.chunks(self.segment_config.num_features)
                        .zip(doc_ids)
                        .for_each(|(vector, doc_id)| {
                            self.insert_for_users(user_ids, *doc_id, vector, op.seq_no)
                                .unwrap();
                        });
                }
                WalOpType::Delete => {
                    debug!("Processing delete operation with seq_no {}", op.seq_no);
                    let doc_ids = op.doc_ids();
                    let user_ids = op.user_ids();
                    assert!(op.data().is_empty());
                    user_ids.iter().try_for_each(|&user_id| {
                        doc_ids.iter().try_for_each(|&doc_id| {
                            self.remove(user_id, doc_id, op.seq_no)?;
                            Ok(())
                        })
                    })?;
                }
            }
            Ok(1)
        } else {
            Ok(0)
        }
    }

    pub fn insert(&self, doc_id: u128, data: &[f32]) -> Result<()> {
        self.mutable_segments
            .read()
            .mutable_segment
            .read()
            .insert(doc_id, data)?;

        // The doc is not immediately searchable, but we update the metrics anyway
        INTERNAL_METRICS.num_searchable_docs_inc(&self.collection_name);

        Ok(())
    }

    pub fn insert_for_users(
        &self,
        user_ids: &[u128],
        doc_id: u128,
        data: &[f32],
        sequence_number: u64,
    ) -> Result<()> {
        for user_id in user_ids {
            self.mutable_segments
                .read()
                .mutable_segment
                .read()
                .insert_for_user(*user_id, doc_id, data, sequence_number)?;
        }

        // The doc is not immediately searchable, but we update the metrics anyway
        INTERNAL_METRICS.num_searchable_docs_inc(&self.collection_name);

        Ok(())
    }

    /// Returns the number of features in the collection
    pub fn dimensions(&self) -> usize {
        self.segment_config.num_features
    }

    /// Turns mutable segment into immutable one, which is the only queryable segment type
    /// currently.
    pub fn flush(&self) -> Result<String> {
        // Try to acquire the flushing lock. If it fails, then another thread is already flushing.
        // This is a best effort approach, and we don't want to block the main thread.
        match self.flushing.try_lock() {
            std::result::Result::Ok(_) => {
                // If there are no documents to flush, just return an empty string (meaning we're not flushing)
                if self
                    .mutable_segments
                    .read()
                    .mutable_segment
                    .read()
                    .num_docs()
                    == 0
                {
                    debug!("No documents to flush");
                    *self.last_flush_time.lock().unwrap() = Instant::now();
                    return Ok(String::new());
                }

                let tmp_name = format!("tmp_segment_{}", rand::random::<u64>());
                let writable_base_directory = format!("{}/{}", self.base_directory, tmp_name);
                {
                    // Grab the write lock and swap tmp_segment with mutable_segment
                    let mut new_writable_segment =
                        MutableSegment::new(self.segment_config.clone(), writable_base_directory)?;

                    let mutable_segments = self.mutable_segments.write();
                    let mut mutable_segment = mutable_segments.mutable_segment.write();
                    std::mem::swap(&mut *mutable_segment, &mut new_writable_segment);

                    let pending_mutable_segment = PendingMutableSegment::new(new_writable_segment);
                    *mutable_segments.pending_mutable_segment.write() =
                        Some(pending_mutable_segment);
                }

                // This is for testing behaviors of invalidating/inserting while being flushed
                #[cfg(test)]
                if std::env::var("TEST_SLOW_FLUSH").is_ok() {
                    std::thread::sleep(std::time::Duration::from_secs(1));
                }

                let name_for_new_segment = format!("segment_{}", rand::random::<u64>());
                let mutable_segments_read = self.mutable_segments.read();
                let pending_segment_read = mutable_segments_read
                    .pending_mutable_segment
                    .upgradable_read();

                let last_sequence_number;
                {
                    let pending_mutable_segment = pending_segment_read.as_ref().unwrap();
                    last_sequence_number = pending_mutable_segment.last_sequence_number();
                    pending_mutable_segment
                        .build(self.base_directory.clone(), name_for_new_segment.clone())?;
                }

                // Read the segment
                let spann_reader = MultiSpannReader::new(format!(
                    "{}/{}",
                    self.base_directory,
                    name_for_new_segment.clone()
                ));
                let index = spann_reader.read::<Q>(
                    self.segment_config.posting_list_encoding_type.clone(),
                    self.segment_config.num_features,
                )?;
                let segment = BoxedImmutableSegment::FinalizedSegment(Arc::new(RwLock::new(
                    ImmutableSegment::new(index, name_for_new_segment.clone()),
                )));

                // Must grab the write lock to prevent further invalidations when applying pending deletions
                {
                    let mut pending_segment_write =
                        RwLockUpgradableReadGuard::upgrade(pending_segment_read);
                    let pending_deletions = pending_segment_write.as_ref().unwrap().deletion_ops();
                    for deletion in pending_deletions {
                        segment.remove(deletion.user_id, deletion.doc_id)?;
                    }
                    *pending_segment_write = None;

                    // Add segments while holding write lock to prevent insertions/invalidations
                    self.add_segments(
                        vec![name_for_new_segment.clone()],
                        vec![segment],
                        last_sequence_number,
                    )?;
                }
                self.trim_wal(last_sequence_number as i64)?;
                *self.last_flush_time.lock().unwrap() = Instant::now();
                Ok(name_for_new_segment)
            }
            Err(_) => {
                return Err(anyhow::anyhow!("Another thread is already flushing"));
            }
        }
    }

    /// Get a consistent snapshot for the collection
    /// TODO(hicder): Get the consistent snapshot w.r.t. time.
    pub fn get_snapshot(self: Arc<Self>) -> Result<Snapshot<Q>> {
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
        let segments: Vec<BoxedImmutableSegment<Q>> = toc
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
        segments: Vec<BoxedImmutableSegment<Q>>,
        last_sequence_number: u64,
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
        let versions_info_read = self.versions_info.upgradable_read();
        let current_version = versions_info_read.current_version;
        let new_version = current_version + 1;

        let mut new_toc = self.versions.get(&current_version).unwrap().toc.clone();
        new_toc.extend_from_slice(&names);
        let num_new_toc_segments = new_toc.len();

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
            sequence_number: last_sequence_number as i64,
        };
        serde_json::to_writer_pretty(&mut tmp_toc_file, &toc)?;

        // Once success, update the current version and ref counts.
        let mut versions_info_write = RwLockUpgradableReadGuard::upgrade(versions_info_read);
        let toc_path = format!("{}/version_{}", self.base_directory, new_version);
        std::fs::rename(tmp_toc_path, toc_path)?;

        versions_info_write.current_version = new_version;
        versions_info_write
            .version_ref_counts
            .insert(new_version, 0);

        self.versions.insert(new_version, toc);

        // New TOC now contains the new segments. Update the metrics
        INTERNAL_METRICS
            .num_active_segments_set(&self.collection_name, num_new_toc_segments as i64);

        Ok(())
    }

    /// Replace old segments with a new segment.
    /// This function is not thread-safe. Caller needs to ensure the thread safety.
    fn replace_segment(
        &self,
        new_segment: BoxedImmutableSegment<Q>,
        old_segment_names: Vec<String>,
        is_pending: bool,
        versions_info_read: RwLockUpgradableReadGuard<RawRwLock, VersionsInfo>,
    ) -> Result<()> {
        // Make sure the old segments are active
        let current_toc = self
            .versions
            .get(&versions_info_read.current_version)
            .unwrap()
            .toc
            .clone();
        for old_segment_name in old_segment_names.iter() {
            if !current_toc.contains(old_segment_name) {
                return Err(anyhow::anyhow!(
                    "Old segment {} is not active",
                    old_segment_name
                ));
            }
        }

        self.all_segments
            .insert(new_segment.name(), new_segment.clone());

        // Under the lock, we do the following:
        // - Increment the current version
        // - Add the new version to the toc, and persist to disk
        // - Insert the new version to the toc
        let current_version = versions_info_read.current_version;
        let new_version = current_version + 1;

        let mut new_toc = self.versions.get(&current_version).unwrap().toc.clone();
        new_toc.retain(|name| !old_segment_names.contains(name));
        new_toc.push(new_segment.name());
        let num_new_toc_segments = new_toc.len();

        let mut new_pending = self.versions.get(&current_version).unwrap().pending.clone();
        let last_sequence_number = self.versions.get(&current_version).unwrap().sequence_number;
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

            // Since we're just replacing the segment, the sequence number should be the same.
            sequence_number: last_sequence_number,
        };
        serde_json::to_writer_pretty(&mut tmp_toc_file, &toc)?;

        let mut versions_info_write = RwLockUpgradableReadGuard::upgrade(versions_info_read);
        let toc_path = format!("{}/version_{}", self.base_directory, new_version);
        std::fs::rename(tmp_toc_path, toc_path)?;

        versions_info_write.current_version = new_version;
        versions_info_write
            .version_ref_counts
            .insert(new_version, 0);

        self.versions.insert(new_version, toc);

        // New TOC now got rid of the old segments and added the new segment. Update the metrics
        INTERNAL_METRICS
            .num_active_segments_set(&self.collection_name, num_new_toc_segments as i64);

        Ok(())
    }

    /// Replace old segments with a new segment.
    /// This function is thread-safe.
    ///
    /// If some of the segment names in old_segment_names are not active, this function will fail.
    pub fn replace_segment_safe(
        &self,
        new_segment: BoxedImmutableSegment<Q>,
        old_segment_names: Vec<String>,
        is_pending: bool,
    ) -> Result<()> {
        let versions_info_read = self.versions_info.upgradable_read();

        self.replace_segment(
            new_segment,
            old_segment_names,
            is_pending,
            versions_info_read,
        )?;
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
        let mut versions_info_write = self.versions_info.write();
        let count = *versions_info_write
            .version_ref_counts
            .get(&version_number)
            .unwrap_or(&0);
        versions_info_write
            .version_ref_counts
            .insert(version_number, count - 1);
    }

    /// This is thread-safe, and will increment the ref count for the version.
    fn get_current_version_and_increment(&self) -> u64 {
        let mut versions_info_write = self.versions_info.write();
        let current_version = versions_info_write.current_version;

        let count = *versions_info_write
            .version_ref_counts
            .get(&current_version)
            .unwrap_or(&0);
        versions_info_write
            .version_ref_counts
            .insert(current_version, count + 1);

        current_version
    }

    pub fn get_all_segment_names(&self) -> Vec<String> {
        self.all_segments
            .iter()
            .map(|pair| pair.key().clone())
            .collect()
    }

    /// Retrieves information about the currently active segments in the collection.
    ///
    /// Returns a `SegmentInfoAndVersion` struct containing a vector of `SegmentInfo`
    /// for each active segment, as well as the current version of the collection.
    pub fn get_active_segment_infos(&self) -> SegmentInfoAndVersion {
        let current_version = self.versions_info.read().current_version;
        let active_segments = self.versions.get(&current_version).unwrap().toc.clone();
        let segment_infos = self
            .all_segments
            .iter()
            .filter(|pair| active_segments.contains(pair.key()))
            .map(|pair| SegmentInfo {
                name: pair.key().clone(),
                size_in_bytes: pair.value().size_in_bytes_immutable_segments(),
                num_docs: pair.value().num_docs(),
            })
            .collect();
        SegmentInfoAndVersion {
            segment_infos,
            version: current_version,
        }
    }

    pub fn init_optimizing(&self, segments: &Vec<String>) -> Result<String> {
        let random_name = format!("pending_segment_{}", rand::random::<u64>());
        let pending_segment_path = format!("{}/{}", self.base_directory, random_name);
        std::fs::create_dir_all(pending_segment_path.clone())?;
        let mut current_segments = Vec::new();
        for segment in segments {
            current_segments.push(self.all_segments.get(segment).unwrap().clone());
        }
        let pending_segment = PendingSegment::<Q>::new(
            current_segments.clone(),
            pending_segment_path,
            self.segment_config.clone(),
        );
        let new_boxed_segment =
            BoxedImmutableSegment::PendingSegment(Arc::new(RwLock::new(pending_segment)));
        self.replace_segment_safe(new_boxed_segment, segments.clone(), true)?;

        Ok(random_name)
    }

    fn pending_to_finalized(
        &self,
        pending_segment: &str,
        versions_info_read: RwLockUpgradableReadGuard<RawRwLock, VersionsInfo>,
    ) -> Result<String> {
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
        let index = MultiSpannReader::new(new_segment_path.clone()).read::<Q>(
            self.segment_config.posting_list_encoding_type.clone(),
            self.segment_config.num_features,
        )?;
        let new_segment = BoxedImmutableSegment::FinalizedSegment(Arc::new(RwLock::new(
            ImmutableSegment::new(index, random_name.clone()),
        )));
        self.replace_segment(
            new_segment,
            vec![pending_segment.to_string()],
            false,
            versions_info_read,
        )?;
        Ok(random_name)
    }

    /// Runs the specified optimizer on a pending segment.
    ///
    /// This function takes a pending segment (created by `init_optimizing`), applies the
    /// optimization logic defined by the `optimizer`, builds the index for the optimized
    /// segment, applies any pending deletions, and then converts the pending segment
    /// into a finalized immutable segment.
    ///
    /// # Arguments
    ///
    /// * `optimizer` - The optimizer implementation to run (e.g., `MergeOptimizer`, `VacuumOptimizer`).
    /// * `pending_segment` - The name of the pending segment to optimize.
    ///
    /// # Returns
    ///
    /// A `Result` containing the name of the newly created finalized segment on success,
    /// or an error if the optimization process fails.
    pub fn run_optimizer(
        &self,
        optimizer: &impl SegmentOptimizer<Q>,
        pending_segment: &str,
    ) -> Result<String> {
        let segment = self
            .all_segments
            .get(pending_segment)
            .unwrap()
            .value()
            .clone();
        if let BoxedImmutableSegment::PendingSegment(pending_segment) = segment {
            let temp_storage_dir;
            {
                let mut pending_segment = pending_segment.upgradable_read();
                temp_storage_dir = pending_segment.temp_invalidated_ids_storage_directory();
                optimizer.optimize(&pending_segment)?;
                pending_segment.build_index()?;
                pending_segment.with_upgraded(|pending_segment_write| {
                    pending_segment_write.apply_pending_deletions()?;
                    pending_segment_write.switch_to_internal_index();
                    Ok(())
                })?;
            }

            // Remove temporary invalidated ids storage from pending segment.
            std::fs::remove_dir_all(&temp_storage_dir)?;
        }

        // Make the pending segment finalized
        // Note that this function will upgrade toc lock.
        let toc_locked = self.versions_info.upgradable_read();
        self.pending_to_finalized(pending_segment, toc_locked)
    }

    fn trim_wal(&self, flushed_seq_no: i64) -> Result<()> {
        if let Some(wal) = &self.wal {
            wal.write().trim_wal(flushed_seq_no)?;
        }
        Ok(())
    }

    pub fn all_segments(&self) -> &DashMap<String, BoxedImmutableSegment<Q>> {
        &self.all_segments
    }

    #[inline(always)]
    fn remove_impl(
        version: &TableOfContent,
        all_segments: &DashMap<String, BoxedImmutableSegment<Q>>,
        mutable_segment: &MutableSegment,
        pending_mutable_segment: Option<&PendingMutableSegment>,
        user_id: u128,
        doc_id: u128,
        sequence_number: u64,
    ) -> Result<()> {
        mutable_segment.invalidate(user_id, doc_id, sequence_number)?;

        if pending_mutable_segment.is_some() {
            pending_mutable_segment
                .unwrap()
                .invalidate(user_id, doc_id)?;
        }

        for segment_name in version.toc.iter() {
            let segment = all_segments.get(segment_name).unwrap();
            segment.remove(user_id, doc_id)?;
        }

        Ok(())
    }

    /// Removes a document with the given `user_id` and `doc_id` from the collection.
    ///
    /// This function marks the document as invalid in the mutable segment and removes it from all immutable segments.
    ///
    /// # Arguments
    ///
    /// * `user_id` - The user ID of the document to remove.
    /// * `doc_id` - The document ID of the document to remove.
    /// * `sequence_number` - The sequence number of the operation.
    pub fn remove(&self, user_id: u128, doc_id: u128, sequence_number: u64) -> Result<()> {
        let version_info_read = self.versions_info.read();
        let current_version = version_info_read.current_version;
        let version = self.versions.get(&current_version).unwrap();

        let mutable_segments = self.mutable_segments.read();
        let mutable_segment = mutable_segments.mutable_segment.read();
        let pending_mutable_segment = mutable_segments.pending_mutable_segment.read();

        Self::remove_impl(
            &version,
            &self.all_segments,
            &mutable_segment,
            pending_mutable_segment.as_ref(),
            user_id,
            doc_id,
            sequence_number,
        )?;

        // The doc is immediately not searchable
        INTERNAL_METRICS.num_searchable_docs_dec(&self.collection_name);

        Ok(())
    }

    /// Iterates through the active segments and initiates vacuuming for segments that exceed the auto-vacuum threshold.
    ///
    /// This function checks each active segment to determine if it should be auto-vacuumed based on its configuration.
    /// If a segment exceeds the threshold, a vacuuming operation is initiated to optimize the segment.
    fn auto_vacuum(&self) -> Result<()> {
        let segment_infos = self.get_active_segment_infos().segment_infos;

        for segment_info in segment_infos.iter() {
            let segment_name = &segment_info.name;
            if let Some(segment) = self.all_segments.get(segment_name) {
                if segment.should_auto_vacuum() {
                    info!(
                        "{}: Auto vacuuming segment {}",
                        self.collection_name, segment_name
                    );
                    let segments_to_optimize = vec![segment_name.clone()];
                    if let Result::Ok(pending_segment) = self.init_optimizing(&segments_to_optimize)
                    {
                        let vacuum_optimizer = VacuumOptimizer::<Q>::new();
                        self.run_optimizer(&vacuum_optimizer, &pending_segment)?;
                    } else {
                        warn!(
                            "{}: Failed to init optimizing segment {}. Skipping",
                            self.collection_name, segment_name
                        );
                    }
                }
            }
        }

        Ok(())
    }

    /// Checks if the number of segments exceeds the maximum allowed and initiates a merge operation.
    ///
    /// If the number of active segments in the collection is greater than the configured
    /// `max_number_of_segments`, this function selects the smallest segments and merges them
    /// into a single segment, reducing the overall number of segments in the collection.
    fn auto_merge(&self) -> Result<()> {
        let mut segment_infos = self.get_active_segment_infos().segment_infos;
        let max_num_segments = self.segment_config.max_number_of_segments;

        if segment_infos.len() <= max_num_segments {
            return Ok(());
        }

        // TODO(hicder): For now, we only merge the smallest segments together until
        // we have less than max_num_segments segments. We should also support having
        // a minimum number of docs per segment, so that segment isn't too small.
        segment_infos.sort_by(|a, b| a.num_docs.cmp(&b.num_docs));
        let num_segments_to_merge = segment_infos.len() - (max_num_segments - 1);

        let segmens_to_merge = segment_infos
            .iter()
            .take(num_segments_to_merge)
            .map(|s| s.name.clone())
            .collect::<Vec<_>>();

        info!(
            "{}: Auto merging segments {:?}",
            self.collection_name, segmens_to_merge
        );

        if let Result::Ok(pending_segment) = self.init_optimizing(&segmens_to_merge) {
            let merge_optimizer = MergeOptimizer::<Q>::new();
            self.run_optimizer(&merge_optimizer, &pending_segment)?;
        } else {
            warn!(
                "{}: Failed to init optimizing segment {:?}. Skipping",
                self.collection_name, segmens_to_merge
            );
        }

        Ok(())
    }

    /// Checks for segments that should be auto-vacuumed and initiates optimization for them.
    pub fn auto_optimize(&self) -> Result<()> {
        info!("{}: Auto optimizing", self.collection_name);
        self.auto_vacuum()?;
        self.auto_merge()?;

        Ok(())
    }
}

// Test
#[cfg(test)]
mod tests {

    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;
    use std::time::Instant;

    use anyhow::{Ok, Result};
    use config::collection::CollectionConfig;
    use metrics::INTERNAL_METRICS;
    use parking_lot::RwLock;
    use quantization::noq::noq::NoQuantizerL2;
    use rand::Rng;
    use tempdir::TempDir;

    use crate::collection::collection::Collection;
    use crate::collection::reader::CollectionReader;
    use crate::collection::snapshot::Snapshot;
    use crate::optimizers::noop::NoopOptimizer;
    use crate::segment::{BoxedImmutableSegment, MockedSegment, Segment};
    use crate::wal::entry::WalOpType;

    #[test]
    fn test_collection() -> Result<()> {
        let collection_name = "test_collection";
        let temp_dir = TempDir::new(collection_name)?;
        let base_directory: String = temp_dir.path().to_str().unwrap().to_string();
        let segment_config = CollectionConfig::default_test_config();
        let collection = Arc::new(
            Collection::<NoQuantizerL2>::new(
                collection_name.to_string(),
                base_directory.clone(),
                segment_config,
            )
            .unwrap(),
        );

        {
            let segment1: BoxedImmutableSegment<NoQuantizerL2> =
                BoxedImmutableSegment::MockedNoQuantizationSegment(Arc::new(RwLock::new(
                    MockedSegment::new("segment1".to_string()),
                )));
            let segment2: BoxedImmutableSegment<NoQuantizerL2> =
                BoxedImmutableSegment::MockedNoQuantizationSegment(Arc::new(RwLock::new(
                    MockedSegment::new("segment2".to_string()),
                )));

            collection
                .add_segments(
                    vec!["segment1".to_string(), "segment2".to_string()],
                    vec![segment1.clone(), segment2.clone()],
                    0,
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
                    0,
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
        let collection_name = "test_collection";
        let temp_dir = TempDir::new(collection_name)?;
        let base_directory: String = temp_dir.path().to_str().unwrap().to_string();
        let segment_config = CollectionConfig::default_test_config();

        let collection = Arc::new(
            Collection::<NoQuantizerL2>::new(
                collection_name.to_string(),
                base_directory.clone(),
                segment_config,
            )
            .unwrap(),
        );
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
                    0,
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

    #[tokio::test]
    async fn test_collection_optimizer() -> Result<()> {
        let collection_name = "test_collection";
        let temp_dir = TempDir::new(collection_name)?;
        let base_directory: String = temp_dir.path().to_str().unwrap().to_string();
        let segment_config = CollectionConfig::default_test_config();
        let collection = Arc::new(
            Collection::<NoQuantizerL2>::new(
                collection_name.to_string(),
                base_directory.clone(),
                segment_config,
            )
            .unwrap(),
        );

        // Add a document and flush
        collection.insert_for_users(&[0], 1, &[1.0, 2.0, 3.0, 4.0], 0)?;
        collection.flush()?;

        let segment_names = collection.get_all_segment_names();
        assert_eq!(segment_names.len(), 1);

        let pending_segment = collection.init_optimizing(&segment_names)?;

        let snapshot = collection.clone().get_snapshot()?;
        let snapshot = Arc::new(snapshot);
        assert_eq!(snapshot.segments.len(), 1);
        assert_eq!(snapshot.version(), 2);
        let segment_name = snapshot.segments[0].name();
        assert_eq!(segment_name, pending_segment);

        let result =
            Snapshot::search_for_users(snapshot, &[0], vec![1.0, 2.0, 3.0, 4.0], 10, 10, false)
                .await
                .unwrap();
        assert_eq!(result.id_with_scores.len(), 1);
        assert_eq!(result.id_with_scores[0].doc_id, 1);

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

    #[tokio::test]
    async fn test_collection_multi_thread_optimizer() -> Result<()> {
        let collection_name = "test_collection";
        let temp_dir = TempDir::new(collection_name)?;
        let base_directory: String = temp_dir.path().to_str().unwrap().to_string();
        let segment_config = CollectionConfig::default_test_config();
        let collection = Arc::new(
            Collection::<NoQuantizerL2>::new(
                collection_name.to_string(),
                base_directory.clone(),
                segment_config,
            )
            .unwrap(),
        );

        collection.insert_for_users(&[0], 1, &[1.0, 2.0, 3.0, 4.0], 0)?;
        collection.flush()?;

        // A thread to optimize the segment
        let collection_cpy_for_optimizer = collection.clone();
        let collection_cpy_for_query = collection.clone();

        let stopped = Arc::new(AtomicBool::new(false));
        let stopped_cpy_for_optimizer = stopped.clone();
        tokio::spawn(async move {
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
        tokio::spawn(async move {
            while !stopped_cpy_for_query.load(std::sync::atomic::Ordering::Relaxed) {
                let c = collection_cpy_for_query.clone();
                let snapshot = c.get_snapshot().unwrap();
                let snapshot = Arc::new(snapshot);
                let result = Snapshot::search_for_users(
                    snapshot,
                    &[0],
                    vec![1.0, 2.0, 3.0, 4.0],
                    10,
                    10,
                    false,
                )
                .await
                .unwrap();

                assert_eq!(result.id_with_scores.len(), 1);
                assert_eq!(result.id_with_scores[0].doc_id, 1);

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

    #[test]
    fn test_collection_reader() -> Result<()> {
        let collection_name = "test_collection";
        let temp_dir = TempDir::new(collection_name)?;
        let base_directory: String = temp_dir.path().to_str().unwrap().to_string();
        let segment_config = CollectionConfig::default_test_config();
        // write the collection config
        let collection_config_path = format!("{}/collection_config.json", base_directory);
        serde_json::to_writer_pretty(
            std::fs::File::create(collection_config_path)?,
            &segment_config,
        )?;

        {
            let collection = Arc::new(
                Collection::<NoQuantizerL2>::new(
                    collection_name.to_string(),
                    base_directory.clone(),
                    segment_config,
                )
                .unwrap(),
            );

            collection.insert_for_users(&[0], 1, &[1.0, 2.0, 3.0, 4.0], 0)?;
            collection.flush()?;

            let segment_names = collection.get_all_segment_names();
            assert_eq!(segment_names.len(), 1);

            let pending_segment = collection.init_optimizing(&segment_names)?;

            let toc = collection.get_current_toc();
            assert_eq!(toc.pending.len(), 1);
            assert_eq!(toc.pending.get(&pending_segment).unwrap().len(), 1);
        }

        let reader = CollectionReader::new(collection_name.to_string(), base_directory);
        let collection = reader.read::<NoQuantizerL2>()?;
        let toc = collection.get_current_toc();
        assert_eq!(toc.pending.len(), 1);
        Ok(())
    }

    #[tokio::test]
    async fn test_collection_with_wal() -> Result<()> {
        let collection_name = "test_collection";
        let temp_dir = TempDir::new(collection_name)?;
        let base_directory: String = temp_dir.path().to_str().unwrap().to_string();
        let mut segment_config = CollectionConfig::default_test_config();
        segment_config.wal_file_size = 1024 * 1024;

        let collection = Arc::new(
            Collection::<NoQuantizerL2>::new(
                collection_name.to_string(),
                base_directory.clone(),
                segment_config,
            )
            .unwrap(),
        );
        collection
            .write_to_wal(&[1], &[0], &[1.0, 2.0, 3.0, 4.0], WalOpType::Insert)
            .await?;
        collection
            .write_to_wal(&[2], &[0], &[1.0, 2.0, 3.0, 4.0], WalOpType::Insert)
            .await?;
        collection
            .write_to_wal(&[3], &[0], &[1.0, 2.0, 3.0, 4.0], WalOpType::Insert)
            .await?;
        collection
            .write_to_wal(&[4], &[0], &[1.0, 2.0, 3.0, 4.0], WalOpType::Insert)
            .await?;
        collection
            .write_to_wal(&[5], &[0], &[1.0, 2.0, 3.0, 4.0], WalOpType::Insert)
            .await?;
        collection
            .write_to_wal(&[5], &[0], &[], WalOpType::Delete)
            .await?;

        // Process all ops
        loop {
            let op = collection.process_one_op().await?;
            if op == 0 {
                break;
            }
        }
        collection.flush()?;
        let segment_names = collection.get_all_segment_names();
        assert_eq!(segment_names.len(), 1);

        let segment_name = segment_names[0].clone();

        let segment = collection
            .all_segments()
            .get(&segment_name)
            .unwrap()
            .value()
            .clone();
        match segment {
            BoxedImmutableSegment::FinalizedSegment(immutable_segment) => {
                assert!(immutable_segment.read().get_point_id(0, 1).is_some());
                assert!(immutable_segment.read().get_point_id(0, 2).is_some());
                assert!(immutable_segment.read().get_point_id(0, 3).is_some());
                assert!(immutable_segment.read().get_point_id(0, 4).is_some());
                assert!(immutable_segment.read().get_point_id(0, 5).is_none());
            }
            _ => {
                assert!(false);
            }
        }
        let toc = collection.get_current_toc();
        assert_eq!(toc.pending.len(), 0);
        assert_eq!(toc.sequence_number, 5);
        Ok(())
    }

    #[tokio::test]
    async fn test_collection_with_wal_reopen() -> Result<()> {
        let collection_name = "test_collection";
        let temp_dir = TempDir::new(collection_name)?;
        let base_directory: String = temp_dir.path().to_str().unwrap().to_string();
        let mut segment_config = CollectionConfig::default_test_config();
        segment_config.wal_file_size = 1024 * 1024;

        Collection::<NoQuantizerL2>::init_new_collection(base_directory.clone(), &segment_config)?;

        // Insert but don't flush
        {
            let reader = CollectionReader::new(collection_name.to_string(), base_directory.clone());
            let collection = reader.read::<NoQuantizerL2>()?;
            collection
                .write_to_wal(&[1], &[0], &[1.0, 2.0, 3.0, 4.0], WalOpType::Insert)
                .await?;
            collection
                .write_to_wal(&[2], &[0], &[1.0, 2.0, 3.0, 4.0], WalOpType::Insert)
                .await?;
            collection
                .write_to_wal(&[1], &[0], &[], WalOpType::Delete)
                .await?;

            // Process all ops
            loop {
                let op = collection.process_one_op().await?;
                if op == 0 {
                    break;
                }
            }
        }

        let segment1_name;
        {
            let reader = CollectionReader::new(collection_name.to_string(), base_directory.clone());
            let collection = reader.read::<NoQuantizerL2>()?;
            let toc = collection.get_current_toc();
            assert_eq!(toc.pending.len(), 0);
            assert_eq!(toc.sequence_number, -1);

            collection.flush()?;
            let segment_names = collection.get_all_segment_names();
            assert_eq!(segment_names.len(), 1);

            segment1_name = segment_names[0].clone();

            let segment1 = collection
                .all_segments()
                .get(&segment1_name)
                .unwrap()
                .value()
                .clone();
            match segment1 {
                BoxedImmutableSegment::FinalizedSegment(immutable_segment) => {
                    assert!(immutable_segment.read().get_point_id(0, 1).is_none());
                    assert!(immutable_segment.read().get_point_id(0, 2).is_some());
                }
                _ => {
                    assert!(false);
                }
            }
            let toc = collection.get_current_toc();
            assert_eq!(toc.pending.len(), 0);
            assert_eq!(toc.sequence_number, 2);

            // Write 2 more ops, but don't flush
            collection
                .write_to_wal(&[2], &[0], &[1.0, 2.0, 3.0, 4.0], WalOpType::Insert)
                .await?;
            collection
                .write_to_wal(&[3], &[0], &[1.0, 2.0, 3.0, 4.0], WalOpType::Insert)
                .await?;
            collection
                .write_to_wal(&[4], &[0], &[1.0, 2.0, 3.0, 4.0], WalOpType::Insert)
                .await?;
            collection
                .write_to_wal(&[1], &[0], &[], WalOpType::Delete)
                .await?;
            collection
                .write_to_wal(&[2], &[0], &[], WalOpType::Delete)
                .await?;
            collection
                .write_to_wal(&[3], &[0], &[], WalOpType::Delete)
                .await?;
        }

        {
            let reader = CollectionReader::new(collection_name.to_string(), base_directory);
            let collection = reader.read::<NoQuantizerL2>()?;
            let toc = collection.get_current_toc();
            assert_eq!(toc.pending.len(), 0);
            assert_eq!(toc.sequence_number, 2);

            collection.flush()?;
            let segment_names = collection.get_all_segment_names();
            assert_eq!(segment_names.len(), 2);

            let segment2_name = if segment1_name == segment_names[0] {
                segment_names[1].clone()
            } else {
                segment_names[0].clone()
            };

            let segment2 = collection
                .all_segments()
                .get(&segment2_name)
                .unwrap()
                .value()
                .clone();
            match segment2 {
                BoxedImmutableSegment::FinalizedSegment(immutable_segment) => {
                    assert!(immutable_segment.read().get_point_id(0, 1).is_none());
                    assert!(immutable_segment.read().get_point_id(0, 2).is_none());
                    assert!(immutable_segment.read().get_point_id(0, 3).is_none());
                }
                _ => {
                    assert!(false);
                }
            }

            let segment1 = collection
                .all_segments()
                .get(&segment1_name)
                .unwrap()
                .value()
                .clone();
            match segment1 {
                BoxedImmutableSegment::FinalizedSegment(immutable_segment) => {
                    assert!(immutable_segment.read().is_invalidated(0, 2)?);
                }
                _ => {
                    assert!(false);
                }
            }
            let toc = collection.get_current_toc();
            assert_eq!(toc.pending.len(), 0);
            assert_eq!(toc.sequence_number, 8);
        }

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_collection_inval_during_flush() -> Result<()> {
        // Set the environment variable to activate the sleep during flush
        std::env::set_var("TEST_SLOW_FLUSH", "1");

        let collection_name = "test_collection_inval_during_flush";
        let temp_dir = TempDir::new(collection_name)?;
        let base_directory: String = temp_dir.path().to_str().unwrap().to_string();
        let segment_config = CollectionConfig::default_test_config();
        let collection = Arc::new(
            Collection::<NoQuantizerL2>::new(
                collection_name.to_string(),
                base_directory.clone(),
                segment_config,
            )
            .unwrap(),
        );

        collection.insert_for_users(&[0], 1, &[1.0, 2.0, 3.0, 4.0], 0)?;
        collection.insert_for_users(&[0], 2, &[2.0, 2.0, 3.0, 4.0], 1)?;
        collection.insert_for_users(&[0], 3, &[3.0, 2.0, 3.0, 4.0], 2)?;
        assert!(collection.remove(0, 2, 3).is_ok());

        let collection_cpy_for_flush = collection.clone();
        let collection_cpy_for_inval = collection.clone();

        // Use a channel to communicate the segment name from the flush thread
        let (tx, rx) = tokio::sync::oneshot::channel();

        // A thread to flush
        tokio::spawn(async move {
            let start = Instant::now();
            println!("Flush thread: Starting flush...");
            let segment_name = collection_cpy_for_flush.flush().unwrap();
            println!("Flush thread: Flush completed in {:?}", start.elapsed());
            tx.send(segment_name).unwrap();
        });

        // A thread to invalidate
        tokio::spawn(async move {
            let start = Instant::now();
            // Do not invalidate too soon
            std::thread::sleep(std::time::Duration::from_millis(1));
            println!("Invalidate thread: Starting invalidation...");
            assert!(collection_cpy_for_inval.remove(0, 1, 4).is_ok());
            println!(
                "Invalidate thread: Invalidation completed in {:?}",
                start.elapsed()
            );
        });

        // Receive the segment name from the flush thread
        let segment_name = rx.await.unwrap();

        let segment = collection
            .all_segments()
            .get(&segment_name)
            .unwrap()
            .value()
            .clone();
        match segment {
            BoxedImmutableSegment::FinalizedSegment(immutable_segment) => {
                assert!(immutable_segment.read().get_point_id(0, 2).is_none());
                assert!(immutable_segment.read().get_point_id(0, 1).is_some());
                assert!(immutable_segment.read().is_invalidated(0, 1)?);
            }
            _ => {}
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_collection_inval_insert_inval_during_flush() -> Result<()> {
        // Set the environment variable to activate the sleep during flush
        std::env::set_var("TEST_SLOW_FLUSH", "1");

        let collection_name = "test_collection_inval_insert_inval_during_flush";
        let temp_dir = TempDir::new(collection_name)?;
        let base_directory: String = temp_dir.path().to_str().unwrap().to_string();
        let segment_config = CollectionConfig::default_test_config();
        let collection = Arc::new(
            Collection::<NoQuantizerL2>::new(
                collection_name.to_string(),
                base_directory.clone(),
                segment_config,
            )
            .unwrap(),
        );

        collection.insert(1, &[1.0, 2.0, 3.0, 4.0])?;
        collection.insert(2, &[2.0, 2.0, 3.0, 4.0])?;
        collection.insert(3, &[3.0, 2.0, 3.0, 4.0])?;

        let collection_cpy_for_flush = collection.clone();
        let collection_cpy_for_inval = collection.clone();

        // Use a channel to communicate the segment name from the flush thread
        let (tx, rx) = tokio::sync::oneshot::channel();

        // A thread to flush
        tokio::spawn(async move {
            let start = Instant::now();
            println!("Flush thread: Starting flush...");
            let segment_name = collection_cpy_for_flush.flush().unwrap();
            println!("Flush thread: Flush completed in {:?}", start.elapsed());
            tx.send(segment_name).unwrap();
        });

        // A thread to invalidate
        tokio::spawn(async move {
            let start = Instant::now();
            // Do not invalidate too soon
            std::thread::sleep(std::time::Duration::from_millis(1));
            println!("Invalidate thread: Starting invalidation...");
            assert!(collection_cpy_for_inval.remove(0, 1, 1).is_ok());
            assert!(collection_cpy_for_inval
                .insert(1, &[1.0, 2.0, 3.0, 4.0])
                .is_ok());
            assert!(collection_cpy_for_inval.remove(0, 1, 2).is_ok());
            assert!(collection_cpy_for_inval
                .insert(4, &[1.0, 2.0, 3.0, 4.0])
                .is_ok());
            assert!(collection_cpy_for_inval.remove(0, 4, 3).is_ok());
            println!(
                "Invalidate thread: Invalidation completed in {:?}",
                start.elapsed()
            );
        });

        // Receive the segment name from the flush thread
        let segment_name = rx.await.unwrap();

        let segment = collection
            .all_segments()
            .get(&segment_name)
            .unwrap()
            .value()
            .clone();
        match segment {
            BoxedImmutableSegment::FinalizedSegment(immutable_segment) => {
                assert!(immutable_segment.read().is_invalidated(0, 1)?);
                assert!(!collection
                    .mutable_segments
                    .read()
                    .mutable_segment
                    .read()
                    .is_valid_doc_id(0, 1));
                assert!(immutable_segment.read().get_point_id(0, 4).is_none());
                assert!(!collection
                    .mutable_segments
                    .read()
                    .mutable_segment
                    .read()
                    .is_valid_doc_id(0, 4));
            }
            _ => {}
        }
        Ok(())
    }

    #[test]
    fn test_collection_inval() {
        let collection_name = "test_collection_inval";
        let temp_dir = TempDir::new(collection_name).expect("Failed to create temporary directory");
        let base_directory: String = temp_dir.path().to_str().unwrap().to_string();
        let segment_config = CollectionConfig::default_test_config();
        let collection = Arc::new(
            Collection::<NoQuantizerL2>::new(
                collection_name.to_string(),
                base_directory.clone(),
                segment_config,
            )
            .unwrap(),
        );

        assert!(collection
            .insert_for_users(&[0], 1, &[1.0, 2.0, 3.0, 4.0], 0)
            .is_ok());
        assert!(collection
            .insert_for_users(&[0], 2, &[2.0, 2.0, 3.0, 4.0], 1)
            .is_ok());
        assert!(collection
            .insert_for_users(&[0], 3, &[3.0, 2.0, 3.0, 4.0], 2)
            .is_ok());
        assert!(collection.remove(0, 2, 3).is_ok());

        assert!(collection.flush().is_ok());
        assert!(collection
            .insert_for_users(&[0], 1, &[1.0, 2.0, 3.0, 4.0], 0)
            .is_ok());
        assert!(collection
            .insert_for_users(&[0], 2, &[1.0, 2.0, 3.0, 4.0], 1)
            .is_ok());
        assert!(collection
            .insert_for_users(&[0], 3, &[2.0, 2.0, 3.0, 4.0], 2)
            .is_ok());
        assert!(collection
            .insert_for_users(&[0], 4, &[3.0, 2.0, 3.0, 4.0], 3)
            .is_ok());
        assert!(collection.remove(0, 2, 4).is_ok());
        assert!(collection.remove(0, 3, 5).is_ok());
        assert!(collection.remove(0, 4, 6).is_ok());

        let segment_names = collection.get_all_segment_names();
        assert_eq!(segment_names.len(), 1);

        let segment_name = segment_names[0].clone();

        let segment = collection
            .all_segments()
            .get(&segment_name)
            .unwrap()
            .value()
            .clone();
        match segment {
            BoxedImmutableSegment::FinalizedSegment(immutable_segment) => {
                assert!(immutable_segment.read().get_point_id(0, 1).is_some());
                assert!(immutable_segment.read().get_point_id(0, 2).is_none());
                assert!(immutable_segment.read().get_point_id(0, 3).is_some());
                assert!(immutable_segment.read().is_invalidated(0, 3).unwrap());

                assert!(collection
                    .mutable_segments
                    .read()
                    .mutable_segment
                    .read()
                    .is_valid_doc_id(0, 1));
                assert!(!collection
                    .mutable_segments
                    .read()
                    .mutable_segment
                    .read()
                    .is_valid_doc_id(0, 2));
                assert!(!collection
                    .mutable_segments
                    .read()
                    .mutable_segment
                    .read()
                    .is_valid_doc_id(0, 3));
                assert!(!collection
                    .mutable_segments
                    .read()
                    .mutable_segment
                    .read()
                    .is_valid_doc_id(0, 4));
            }
            _ => {
                assert!(false);
            }
        }
    }

    /// Tests invalidation logic with a larger number of documents.
    ///
    /// This test inserts 20 documents into the collection, then removes 5 of them.
    /// It verifies that the removed documents are marked as invalid in the mutable segment
    /// and that the remaining documents are still valid.
    #[test]
    fn test_collection_inval_large() {
        let collection_name = "test_collection_inval_large";
        let temp_dir = TempDir::new(collection_name).expect("Failed to create temporary directory");
        let base_directory: String = temp_dir.path().to_str().unwrap().to_string();
        let segment_config = CollectionConfig::default_test_config();
        let collection = Arc::new(
            Collection::<NoQuantizerL2>::new(
                collection_name.to_string(),
                base_directory.clone(),
                segment_config,
            )
            .unwrap(),
        );

        let num_docs_to_add = 20;
        let num_docs_to_delete = 5;
        let user_id = 0;
        let vector: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0]; // Dummy vector

        // Add 20 data points
        for i in 0..num_docs_to_add {
            assert!(collection
                .insert_for_users(&[user_id], i as u128, &vector, i as u64)
                .is_ok());
        }

        // Delete 5 data points (e.g., doc_ids 0 to 4)
        for i in 0..num_docs_to_delete {
            assert!(collection
                .remove(user_id, i as u128, (num_docs_to_add + i) as u64)
                .is_ok());
        }

        // Verify that the 5 deleted documents are invalidated
        for i in 0..num_docs_to_delete {
            assert!(!collection
                .mutable_segments
                .read()
                .mutable_segment
                .read()
                .is_valid_doc_id(user_id, i as u128));
        }

        // Verify that the remaining 15 documents are still valid
        for i in num_docs_to_delete..num_docs_to_add {
            assert!(collection
                .mutable_segments
                .read()
                .mutable_segment
                .read()
                .is_valid_doc_id(user_id, i as u128));
        }
    }

    #[test]
    fn test_collection_metrics() {
        let collection_name = "test_collection_metrics";
        let temp_dir = TempDir::new(collection_name).unwrap();
        let base_directory: String = temp_dir.path().to_str().unwrap().to_string();
        let mut segment_config = CollectionConfig::default();
        segment_config.num_features = 16; // Set a specific feature size for precise calculations
        let collection = Arc::new(
            Collection::<NoQuantizerL2>::new(
                collection_name.to_string(),
                base_directory.clone(),
                segment_config.clone(),
            )
            .unwrap(),
        );
        let collection_name = &collection.collection_name;

        // Helper functions to get metrics
        let get_active_segments = || INTERNAL_METRICS.num_active_segments_get(collection_name);
        let get_searchable_docs = || INTERNAL_METRICS.num_searchable_docs_get(collection_name);

        // Initial state checks
        assert_eq!(get_searchable_docs(), 0);
        assert_eq!(get_active_segments(), 0);

        // Prepare test data
        let num_features = segment_config.num_features;
        let test_vectors: Vec<f32> = (0..num_features * 5).map(|i| i as f32).collect();
        let doc_ids: Vec<u128> = (0..5).map(|i| i as u128).collect();
        let user_ids: Vec<u128> = vec![42; 5];

        // Insert 5 documents for all users
        println!("Inserting 5 documents...");
        for (idx, &doc_id) in doc_ids.iter().enumerate() {
            let vector = &test_vectors[idx * num_features..(idx + 1) * num_features];
            collection
                .insert_for_users(&user_ids, doc_id, vector, 0)
                .unwrap();
        }

        // Verify metrics after insertions
        assert_eq!(get_searchable_docs(), 5);
        assert_eq!(get_active_segments(), 0);

        // Flush the mutable segment
        println!("Flushing the collection once...");
        assert!(collection.flush().is_ok());
        assert_eq!(
            collection
                .mutable_segments
                .read()
                .mutable_segment
                .read()
                .num_docs(),
            0
        );

        // Verify metrics after flush
        assert_eq!(get_searchable_docs(), 5);
        assert_eq!(get_active_segments(), 1);

        // Remove 2 documents for first user
        println!("Removing 2 documents...");
        let docs_to_delete = &doc_ids[0..2];
        for &doc_id in docs_to_delete {
            collection.remove(user_ids[0], doc_id, 0).unwrap();
        }

        // Verify metrics after document removal
        assert_eq!(get_searchable_docs(), 3);
        assert_eq!(get_active_segments(), 1);

        // Flush again. Should not create a new segment
        println!("Flushing the collection again...");
        assert!(collection.flush().is_ok());
        assert_eq!(get_active_segments(), 1);

        // Insert 2 documents for all users
        println!("Inserting the 2 documents back...");
        let docs_to_insert = docs_to_delete;
        for (idx, &doc_id) in docs_to_insert.iter().enumerate() {
            let vector = &test_vectors[idx * num_features..(idx + 1) * num_features];
            collection
                .insert_for_users(&user_ids, doc_id, vector, 0)
                .unwrap();
        }

        // Flush again to create another immutable segment
        println!("Flushing the collection again...");
        assert!(collection.flush().is_ok());

        // Final metrics verification
        assert_eq!(get_searchable_docs(), 5);
        assert_eq!(get_active_segments(), 2);
    }
}
