use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Ok, Result};
use async_lock::{RwLock, RwLockUpgradableReadGuard};
use atomic_refcell::AtomicRefCell;
use config::collection::CollectionConfig;
use dashmap::DashMap;
use fs_extra::dir::CopyOptions;
use log::{debug, info, warn};
use metrics::INTERNAL_METRICS;
use proto::muopdb::DocumentAttribute;
use quantization::quantization::Quantizer;
use tokio::sync::mpsc::{self, Receiver, Sender};
use tokio::sync::{oneshot, Mutex as AsyncMutex};
use utils::file_io::env::Env;

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

/// Group of args to pass to `Wal::append` method
struct AppendArgs {
    doc_ids: Arc<[u128]>,
    user_ids: Arc<[u128]>,
    op_type: WalOpType<Arc<[f32]>>,
    document_attribute: Option<Arc<Vec<DocumentAttribute>>>,
}

/// Represents an entry for a follower task
struct GroupEntry {
    /// follower's args
    args: AppendArgs,
    /// sender to send the returned sequence number from calling `Wal::append` to the corresponding follower
    seq_tx: oneshot::Sender<u64>,
}

/// Represents a group that the leader will own and use to append to WAL and sync WAl
struct WalWriteGroup {
    /// maximum number of followers in a write group
    max_num_entries: usize,
    /// the followers' contents
    entries: Vec<GroupEntry>,
}

impl WalWriteGroup {
    fn new(max_num_entries: usize) -> Self {
        Self {
            entries: vec![],
            max_num_entries,
        }
    }

    /// Determines whether this group should be closed for processing
    fn should_close(&self) -> bool {
        self.entries.len() >= self.max_num_entries
    }
}

/// Coordinator to keep the current open `WalWriteGroup`
struct WalWriteCoordinator {
    /// The current open group. If None, the next WAL-write task will create a new group
    pub current_group: Option<WalWriteGroup>,
    wal_write_group_size: usize,
}

impl WalWriteCoordinator {
    fn new(wal_write_group_size: usize) -> Self {
        Self {
            current_group: None,
            wal_write_group_size,
        }
    }

    fn new_wal_write_group(&self) -> WalWriteGroup {
        WalWriteGroup::new(self.wal_write_group_size)
    }
}

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

/// The central coordinator for a vector collection, managing the lifecycle of segments,
/// concurrency, and data persistence.
///
/// ### Segment Management
/// A collection consists of three types of segments:
/// - **Mutable Segment**: An in-memory, writable segment where new data is initially inserted.
/// - **Pending Mutable Segment**: A segment currently being flushed to disk. It is searchable
///   but read-only for new inserts.
/// - **Immutable Segments**: Read-only, memory-mapped segments on disk that have been finalized.
///
/// Segments are created during:
/// 1. **Initialization**: A fresh mutable segment is created when the collection starts.
/// 2. **Flush**: The current mutable segment is swapped with a new one and then built into an
///    immutable segment on disk.
/// 3. **Optimization**: Multiple immutable segments are merged or vacuumed into a new finalized
///    segment to improve search performance and reclaim space.
///
/// ### Atomic Versioning and Segment Addition
/// Collection state is managed through a versioning system to ensure consistent reads and atomic
/// updates:
/// - A `TableOfContent` (TOC) defines the exact set of active segments for any given version.
/// - When new segments are added (via `add_segments` or `flush`), the collection creates a new
///   version by writing a new TOC file to disk and renaming it atomically.
/// - `Snapshot`s allow queries to pin a specific version, ensuring that the segments they are
///   searching remain valid even if the collection is modified or optimized.
///
/// ### Row Invalidation (Deletions)
/// Deletions are handled by marking rows as invalid across all relevant segments:
/// - In **Mutable Segments**, rows are invalidated in-memory immediately.
/// - In **Immutable Segments**, invalidations are tracked via on-disk structures (e.g., bitsets).
/// - During a **Flush**, any invalidations that occurred while the segment was being built are
///   replayed to ensure the newly created immutable segment is consistent with the latest state.
///
/// ### Locking Strategy
/// The collection employs a multi-layered locking strategy to maximize concurrency:
/// - **Version Locking**: `versions_info` uses a `RwLock` to manage version transitions and
///   reference counting.
///   - Grab a **write lock** when updating the current version or modifying reference counts
///     (e.g., when creating a snapshot).
///   - Grab a **read lock** when you only need to know the current version number (e.g., during
///     a deletion).
/// - **Mutable Segment Locking**: `mutable_segments` uses a nested `RwLock` approach.
///   - **Read Lock**: Grabbed by search, insert, and invalidate operations to access the current
///     mutable and pending segments.
///   - **Write Lock**: Grabbed during a flush to atomically swap the active mutable segment.
/// - **WAL Locking**: If enabled, `wal` uses a `RwLock` to serialize writes.
///   - Grab a **write lock** when appending new operations to the WAL.
///   - Grab a **read lock** when syncing the WAL to disk.
/// - **Segment Access**: `all_segments` uses a `DashMap` for efficient, concurrent access to
///   immutable segments by name.
/// - **Flush Coordination**: A dedicated `flushing` Mutex ensures that only one background
///   flush operation occurs at a time.
pub struct Collection<Q: Quantizer + Clone + Send + Sync> {
    versions: DashMap<u64, TableOfContent>,
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
    flushing: tokio::sync::Mutex<()>,

    write_coordinator: Arc<AsyncMutex<WalWriteCoordinator>>,

    // Optional env for async I/O operations
    env: Option<Arc<Box<dyn Env>>>,
}

impl<Q: Quantizer + Clone + Send + Sync + 'static> Collection<Q> {
    pub fn config(&self) -> &CollectionConfig {
        &self.segment_config
    }

    /// Creates a new `Collection` instance with the given name, base directory, and configuration.
    ///
    /// This initializes a fresh collection with an empty version (version 0) and a new mutable segment.
    /// If WAL is enabled in the configuration, it will also open or create the WAL.
    pub fn new(
        collection_name: String,
        base_directory: String,
        segment_config: CollectionConfig,
    ) -> Result<Self> {
        let versions: DashMap<u64, TableOfContent> = DashMap::new();
        versions.insert(0, TableOfContent::new(vec![]));

        // Create a new segment_config with a random name
        let random_name = format!("tmp_segment_{}", rand::random::<u64>());
        let segment_base_directory = format!("{base_directory}/{random_name}");

        let mutable_segment = RwLock::new(MutableSegment::new(
            segment_config.clone(),
            segment_base_directory,
        )?);

        let wal = if segment_config.wal_file_size > 0 {
            let wal_directory = format!("{base_directory}/wal");
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
        let wal_write_group_size = segment_config.wal_write_group_size;

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
            flushing: tokio::sync::Mutex::new(()),
            wal,
            sender,
            receiver,
            last_flush_time: Mutex::new(Instant::now()),
            write_coordinator: Arc::new(AsyncMutex::new(WalWriteCoordinator::new(
                wal_write_group_size,
            ))),
            env: None,
        })
    }

    /// Initializes the disk storage for a new collection at the specified base directory.
    ///
    /// This creates the necessary directory structure, writes the initial version 0 metadata,
    /// and saves the collection configuration.
    pub fn init_new_collection(base_directory: String, config: &CollectionConfig) -> Result<()> {
        std::fs::create_dir_all(base_directory.clone())?;

        // Write version 0
        let toc_path = format!("{base_directory}/version_0");
        let toc = TableOfContent::default();
        serde_json::to_writer_pretty(std::fs::File::create(toc_path)?, &toc)?;

        // Write the config file
        let config_path = format!("{base_directory}/collection_config.json");
        serde_json::to_writer_pretty(std::fs::File::create(config_path)?, config)?;

        Ok(())
    }

    /// Reconstructs a `Collection` from existing on-disk state.
    ///
    /// This function reloads the specified version, attaches the provided segments,
    /// and replays any operations from the WAL that occurred after the last flush.
    pub async fn init_from(
        collection_name: String,
        base_directory: String,
        version: u64,
        toc: TableOfContent,
        segments: Vec<BoxedImmutableSegment<Q>>,
        segment_config: CollectionConfig,
        env: Option<Arc<Box<dyn Env>>>,
    ) -> Result<Self> {
        let versions_info = RwLock::new(VersionsInfo::new());
        versions_info.write().await.current_version = version;
        versions_info
            .write()
            .await
            .version_ref_counts
            .insert(version, 0);

        let all_segments = DashMap::new();
        toc.toc.iter().zip(segments).for_each(|(name, segment)| {
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
        let random_base_directory = format!("{base_directory}/{random_name}");
        std::fs::create_dir_all(&random_base_directory)?;
        let mutable_segment = MutableSegment::new(segment_config.clone(), random_base_directory)?;

        let wal = if segment_config.wal_file_size > 0 {
            let wal_directory = format!("{base_directory}/wal");
            let wal = Wal::open(
                &wal_directory,
                segment_config.wal_file_size,
                last_sequence_number,
            )?;
            let iterators = wal.get_iterators();
            let mut seq_no = last_sequence_number + 1;
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
                        debug!("Processing op with seq_no: {entry_seq_no}");
                        let wal_entry = wal_entry.decode(segment_config.num_features);
                        match wal_entry.op_type {
                            WalOpType::Insert(data) => {
                                let doc_ids = wal_entry.doc_ids;
                                let user_ids = wal_entry.user_ids;

                                // Should be safe to unwrap since we only insert
                                let attributes = wal_entry.attributes.unwrap();

                                let mut num_successful_inserts: i64 = 0;
                                data.chunks(segment_config.num_features)
                                    .zip(doc_ids)
                                    .zip(attributes.iter())
                                    .for_each(|((vector, doc_id), attr)| {
                                        for user_id in user_ids {
                                            mutable_segment
                                                .insert_for_user(
                                                    *user_id,
                                                    *doc_id,
                                                    vector,
                                                    entry_seq_no,
                                                    attr.clone(),
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
                                let ver = versions.get(&version).unwrap();

                                let mut num_successful_deletes: i64 = 0;
                                for &user_id in user_ids.iter() {
                                    for &doc_id in doc_ids.iter() {
                                        Self::remove_impl(
                                            &ver,
                                            &all_segments,
                                            &mutable_segment,
                                            /* pending_mutable_segment */ None,
                                            user_id,
                                            doc_id,
                                            entry_seq_no,
                                        )
                                        .await?;

                                        num_successful_deletes += 1;
                                    }
                                }

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
        let wal_write_group_size = segment_config.wal_write_group_size;

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
            flushing: tokio::sync::Mutex::new(()),
            wal,
            sender,
            receiver,
            last_flush_time: Mutex::new(Instant::now()),
            write_coordinator: Arc::new(AsyncMutex::new(WalWriteCoordinator::new(
                wal_write_group_size,
            ))),
            env,
        })
    }

    /// Returns `true` if Write-Ahead Logging (WAL) is enabled for this collection.
    pub fn use_wal(&self) -> bool {
        self.wal.is_some()
    }

    /// Determines whether the collection should perform an automatic flush.
    ///
    /// This check is based on the configured `max_pending_ops` and `max_time_to_flush_ms`.
    /// Returns `true` if either threshold has been reached.
    pub async fn should_auto_flush(&self) -> bool {
        if self.segment_config.max_pending_ops == 0 && self.segment_config.max_time_to_flush_ms == 0
        {
            return false;
        }

        if self.segment_config.max_pending_ops > 0 {
            let current_seq_no = self
                .mutable_segments
                .read()
                .await
                .mutable_segment
                .read()
                .await
                .last_sequence_number();
            let flushed_seq_no = self
                .versions
                .get(&self.current_version().await)
                .unwrap()
                .sequence_number;
            if current_seq_no as i64 - flushed_seq_no >= self.segment_config.max_pending_ops as i64
            {
                debug!(
                    "Flushing because of max pending ops: {}",
                    current_seq_no as i64 - flushed_seq_no
                );
                return true;
            }
        }

        if self.segment_config.max_time_to_flush_ms > 0 {
            let last_flush_time = *self.last_flush_time.lock().unwrap();
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

    /// Writes a batch of operations to the WAL and queues them for processing.
    ///
    /// This method uses a group-commit strategy to batch multiple concurrent WAL writes
    /// for improved performance. Once persisted to the WAL, operations are sent to
    /// an internal channel for asynchronous application to the memory segments.
    pub async fn write_to_wal(
        &self,
        doc_ids: Arc<[u128]>,
        user_ids: Arc<[u128]>,
        wal_op_type: WalOpType<Arc<[f32]>>,
        document_attributes: Option<Arc<Vec<DocumentAttribute>>>,
    ) -> Result<u64> {
        if let Some(wal) = &self.wal {
            // Lock the write coordinator and take out the current group
            let mut coordinator = self.write_coordinator.lock().await;
            let mut current_group = match coordinator.current_group.take() {
                Some(group) => group,
                // Create a new group if current group is None
                None => coordinator.new_wal_write_group(),
            };

            // A closure to append to WAL and send op to channel
            #[allow(clippy::await_holding_lock)]
            let append_wal = async |wal: &RwLock<Wal>, args: AppendArgs| -> Result<u64> {
                // Convert Arc<[f32]> to &[f32] in the enum
                let wal_op_type_ref = match &args.op_type {
                    WalOpType::Insert(data) => WalOpType::Insert(data.as_ref()),
                    WalOpType::Delete => WalOpType::Delete,
                };
                // Write to WAL, and persist to disk.
                // Intentionally keep the write lock until we send the message to the channel.
                // This ensures that message in the channel is the same order as WAL.
                let mut wal_write = wal.write().await;
                let attr_ref = args.document_attribute;
                let seq_no = wal_write.append(
                    &args.doc_ids,
                    &args.user_ids,
                    wal_op_type_ref,
                    attr_ref.clone(),
                )?;

                // Once the WAL is written, send the op to the channel
                let attributes = if let Some(attr_ref) = attr_ref.clone() {
                    let mut attributes = vec![];
                    for attr in attr_ref.iter() {
                        attributes.push(attr.clone());
                    }
                    attributes
                } else {
                    let mut attributes = vec![];
                    for _ in 0..args.doc_ids.len() {
                        attributes.push(DocumentAttribute::default());
                    }
                    attributes
                };
                let op_channel_entry = OpChannelEntry::new(
                    &args.doc_ids,
                    &args.user_ids,
                    seq_no,
                    args.op_type,
                    attributes,
                );
                self.sender.send(op_channel_entry).await?;
                Ok(seq_no)
            };

            // A closure to run as the leader
            let write_follower_entries = |group: WalWriteGroup, leader_args: Option<AppendArgs>| async move {
                info!("Writing {} append entries to WAL", &group.entries.len());

                // Go through all entries in the group and collect results
                let mut results: Vec<(u64, oneshot::Sender<u64>)> = vec![];
                for entry in group.entries {
                    let seq_no = append_wal(wal, entry.args).await?;
                    results.push((seq_no, entry.seq_tx));
                }

                let leader_seq_no = match leader_args {
                    Some(args) => append_wal(wal, args).await?,
                    None => 0, // dummy value, not used
                };

                // Sync the WAL at once
                let entries_synced = self.sync_wal().await?;
                info!("Synced {entries_synced} entries to WAL");

                // Now send seq_no to all followers
                for (seq_no, seq_tx) in results {
                    debug!("[WAL leader] sending seq_no {seq_no} to follower");
                    seq_tx
                        .send(seq_no)
                        .expect("WAL Leader: follower's receiver dropped");
                }

                Ok(leader_seq_no)
            };

            if current_group.should_close() {
                debug!(
                    "[LEADER] group is closing. current size: {}",
                    current_group.entries.len()
                );
                // This task will be the leader. Own the current group and put a new group in the coordinator.
                coordinator.current_group = Some(coordinator.new_wal_write_group());
                drop(coordinator);

                // Write all follower entries to WAL, including this leader's entry
                let leader_seq_no = write_follower_entries(
                    current_group,
                    Some(AppendArgs {
                        doc_ids,
                        user_ids,
                        op_type: wal_op_type,
                        document_attribute: document_attributes,
                    }),
                )
                .await?;

                Ok(leader_seq_no)
            } else {
                debug!(
                    "[FOLLOWER] joining existing group. current size: {}",
                    current_group.entries.len()
                );
                // This task will be the follower and join the current group.
                let (seq_tx, mut seq_rx) = oneshot::channel();

                // Get the 0-based index of this entry in the group
                let follower_entry_id = current_group.entries.len();

                // Create a new entry in the current group
                current_group.entries.push(GroupEntry {
                    args: AppendArgs {
                        doc_ids,
                        user_ids,
                        op_type: wal_op_type,
                        document_attribute: document_attributes,
                    },
                    seq_tx,
                });

                // Put the current group back to the coordinator
                coordinator.current_group = Some(current_group);
                drop(coordinator);

                debug!("[FOLLOWER] Waiting for leader to write to WAL");

                // Wait for either leader completion OR timeout
                tokio::select! {
                    result = &mut seq_rx => {
                        // Normal case: leader processed us
                        let follower_seq_no = result.expect("WAL follower: leader's sender dropped");
                        debug!("[WAL follower] Received seq_no {follower_seq_no} from leader");
                        Ok(follower_seq_no)
                    }
                    _ = tokio::time::sleep(Duration::from_millis(10)) => {
                        debug!("[WAL follower] Timeout reached, checking if I should become leader");

                        // Timeout: check if I'm the first entry and should become leader
                        let mut coordinator = self.write_coordinator.lock().await;

                        if let Some(group) = coordinator.current_group.take() {
                            // Check if I'm the first added entry (index 0)
                            let is_first_follower = follower_entry_id == 0;

                            if is_first_follower {
                                debug!("[WAL follower] I'm the first entry, becoming timeout leader");

                                // I'm the first entry - become leader!
                                coordinator.current_group = Some(coordinator.new_wal_write_group());
                                drop(coordinator);

                                // Run leader job with the timed-out group
                                write_follower_entries(group, None).await?;

                                // Get the seq_no for myself
                                let leader_seq_no = seq_rx.await.expect("Should receive my own seq_no");

                                // Sync the WAL at once
                                let entries_synced = self.sync_wal().await?;
                                info!("[WAL timeout leader] Synced {entries_synced} entries to WAL");

                                debug!("[WAL timeout leader] Returning my own seq_no {leader_seq_no}");
                                Ok(leader_seq_no)
                            } else {
                                debug!("[WAL timeout follower] Not the first entry, putting group back and continuing to wait");

                                // Not the last entry, put group back and continue waiting
                                coordinator.current_group = Some(group);
                                drop(coordinator);

                                // Continue waiting for the actual leader or timeout leader
                                let follower_seq_no = seq_rx.await.expect("WAL timeout follower: leader's sender dropped");
                                debug!("[WAL timeout follower] Finally received seq_no {follower_seq_no} from leader");
                                Ok(follower_seq_no)
                            }
                        } else {
                            // Group was already taken by another leader, wait for result
                            drop(coordinator);
                            let follower_seq_no = seq_rx.await.expect("WAL follower: leader's sender dropped");
                            debug!("[WAL timeout follower] Received seq_no {follower_seq_no} from leader after timeout");
                            Ok(follower_seq_no)
                        }
                    }
                }
            }
        } else {
            Err(anyhow::anyhow!("WAL is not enabled"))
        }
    }

    /// Synchronizes all unpersisted WAL entries to disk.
    ///
    /// Returns the number of entries that were successfully synced.
    pub async fn sync_wal(&self) -> Result<u64> {
        if let Some(wal) = &self.wal {
            let wal_read = wal.read().await;
            let entries_synced = wal_read.sync()?;
            Ok(entries_synced)
        } else {
            Err(anyhow::anyhow!("WAL is not enabled"))
        }
    }

    /// Processes a single pending operation from the internal command channel.
    ///
    /// This applies the operation (insert or delete) to the active segments.
    /// Returns `Ok(1)` if an operation was processed, or `Ok(0)` if the channel was empty.
    pub async fn process_one_op(&self) -> Result<usize> {
        if let std::result::Result::Ok(op) = self.receiver.borrow_mut().try_recv() {
            match op.op_type {
                WalOpType::Insert(()) => {
                    info!("Processing insert operation with seq_no {}", op.seq_no);
                    let doc_ids = op.doc_ids();
                    let user_ids = op.user_ids();
                    let data = op.data();
                    let attributes = op.attributes();
                    for ((vector, doc_id), attr) in data
                        .chunks(self.segment_config.num_features)
                        .zip(doc_ids)
                        .zip(attributes)
                    {
                        self.insert_for_users(user_ids, *doc_id, vector, op.seq_no, attr.clone())
                            .await
                            .unwrap();
                    }
                }
                WalOpType::Delete => {
                    info!("Processing delete operation with seq_no {}", op.seq_no);
                    let doc_ids = op.doc_ids();
                    let user_ids = op.user_ids();
                    assert!(op.data().is_empty());
                    for &user_id in user_ids.iter() {
                        for &doc_id in doc_ids.iter() {
                            self.remove(user_id, doc_id, op.seq_no).await?;
                        }
                    }
                }
            }
            Ok(1)
        } else {
            Ok(0)
        }
    }

    /// Inserts a document into the current mutable segment.
    ///
    /// This is a low-level insertion that bypasses the WAL and multi-user logic.
    /// Primarily used for tests or internal initialization.
    pub async fn insert(&self, doc_id: u128, data: &[f32]) -> Result<()> {
        self.mutable_segments
            .read()
            .await
            .mutable_segment
            .read()
            .await
            .insert(doc_id, data)?;

        // The doc is not immediately searchable, but we update the metrics anyway
        INTERNAL_METRICS.num_searchable_docs_inc(&self.collection_name);

        Ok(())
    }

    /// Inserts a document for a set of users into the current mutable segment.
    ///
    /// This applies the insertion to the active mutable segment, ensuring the document
    /// is associated with each of the provided user IDs and includes specified attributes.
    pub async fn insert_for_users(
        &self,
        user_ids: &[u128],
        doc_id: u128,
        data: &[f32],
        sequence_number: u64,
        document_attribute: DocumentAttribute,
    ) -> Result<()> {
        for user_id in user_ids {
            self.mutable_segments
                .read()
                .await
                .mutable_segment
                .read()
                .await
                .insert_for_user(
                    *user_id,
                    doc_id,
                    data,
                    sequence_number,
                    document_attribute.clone(),
                )?;
        }

        // The doc is not immediately searchable, but we update the metrics anyway
        INTERNAL_METRICS.num_searchable_docs_inc(&self.collection_name);

        Ok(())
    }

    /// Returns the number of dimensions (features) for vectors in this collection.
    pub fn dimensions(&self) -> usize {
        self.segment_config.num_features
    }

    /// Flushes the current mutable segment to disk, creating a new immutable segment.
    ///
    /// This process involves:
    /// 1. Swapping the active mutable segment with a fresh one.
    /// 2. Building the old mutable segment into an on-disk finalized segment.
    /// 3. Replaying any deletions that occurred during the build process.
    /// 4. Atomically updating the collection version to include the new segment.
    /// 5. Trimming the WAL up to the flushed sequence number.
    ///
    /// Returns the name of the newly created segment on success.
    pub async fn flush(&self) -> Result<String> {
        // Try to acquire the flushing lock. If it fails, then another thread is already flushing.
        // This is a best effort approach, and we don't want to block the main thread.
        let guard = match self.flushing.try_lock() {
            std::result::Result::Ok(guard) => guard,
            Err(_) => return Err(anyhow::anyhow!("Another thread is already flushing")),
        };

        // If there are no documents to flush, just return an empty string (meaning we're not flushing)
        if self
            .mutable_segments
            .read()
            .await
            .mutable_segment
            .read()
            .await
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

            let mutable_segments = self.mutable_segments.write().await;
            let mut mutable_segment = mutable_segments.mutable_segment.write().await;
            std::mem::swap(&mut *mutable_segment, &mut new_writable_segment);

            let pending_mutable_segment = PendingMutableSegment::new(new_writable_segment);
            *mutable_segments.pending_mutable_segment.write().await = Some(pending_mutable_segment);
        }

        // This is for testing behaviors of invalidating/inserting while being flushed
        #[cfg(test)]
        if std::env::var("TEST_SLOW_FLUSH").is_ok() {
            std::thread::sleep(std::time::Duration::from_secs(1));
        }

        let name_for_new_segment = format!("segment_{}", rand::random::<u64>());
        let mutable_segments_read = self.mutable_segments.read().await;
        let pending_segment_read = mutable_segments_read
            .pending_mutable_segment
            .upgradable_read()
            .await;

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
        let index = spann_reader
            .read::<Q>(
                self.segment_config.posting_list_encoding_type.clone(),
                self.segment_config.num_features,
            )
            .await?;
        let terms_path = format!(
            "{}/{}/terms",
            self.base_directory,
            name_for_new_segment.clone()
        );
        std::fs::create_dir_all(&terms_path).ok();
        let segment = BoxedImmutableSegment::FinalizedSegment(Arc::new(RwLock::new(
            ImmutableSegment::new(index, name_for_new_segment.clone(), Some(terms_path)),
        )));

        // Must grab the write lock to prevent further invalidations when applying pending deletions
        {
            let mut pending_segment_write =
                RwLockUpgradableReadGuard::upgrade(pending_segment_read).await;
            let pending_deletions = pending_segment_write.as_ref().unwrap().deletion_ops();
            for deletion in pending_deletions {
                segment.remove(deletion.user_id, deletion.doc_id).await?;
            }
            *pending_segment_write = None;

            // Add segments while holding write lock to prevent insertions/invalidations
            self.add_segments(
                vec![name_for_new_segment.clone()],
                vec![segment],
                last_sequence_number,
            )
            .await?;
        }
        self.trim_wal(last_sequence_number as i64).await?;
        *self.last_flush_time.lock().unwrap() = Instant::now();
        drop(guard);
        Ok(name_for_new_segment)
    }

    /// Get a consistent snapshot for the collection
    /// TODO(hicder): Get the consistent snapshot w.r.t. time.
    pub async fn get_snapshot(self: Arc<Self>) -> Result<Snapshot<Q>> {
        if self.versions.is_empty() {
            return Err(anyhow::anyhow!("Collection is empty"));
        }

        let current_version_number = self.get_current_version_and_increment().await;
        let latest_version = self.versions.get(&current_version_number);
        if latest_version.is_none() {
            // It shouldn't happen, but just in case, we still release the version
            self.release_version(current_version_number).await;
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

    /// Returns the current Table of Content, which lists all active segments for the latest version.
    pub async fn get_current_toc(&self) -> TableOfContent {
        self.versions
            .get(&self.current_version().await)
            .unwrap()
            .value()
            .clone()
    }

    /// Add segments to the collection, effectively creating a new version.
    pub async fn add_segments(
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
        let versions_info_read = self.versions_info.upgradable_read().await;
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
        let mut versions_info_write = RwLockUpgradableReadGuard::upgrade(versions_info_read).await;
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

    /// Replaces a set of existing segments with a new segment.
    ///
    /// This is an internal helper that manages the atomic transition to a new collection version
    /// where the old segments are removed and the new one is added.
    ///
    /// This function is not thread-safe. Caller needs to ensure the thread safety.
    async fn replace_segment(
        &self,
        new_segment: BoxedImmutableSegment<Q>,
        old_segment_names: Vec<String>,
        is_pending: bool,
        versions_info_read: RwLockUpgradableReadGuard<'_, VersionsInfo>,
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
            .insert(new_segment.name().await, new_segment.clone());

        // Under the lock, we do the following:
        // - Increment the current version
        // - Add the new version to the toc, and persist to disk
        // - Insert the new version to the toc
        let current_version = versions_info_read.current_version;
        let new_version = current_version + 1;

        let mut new_toc = self.versions.get(&current_version).unwrap().toc.clone();
        new_toc.retain(|name| !old_segment_names.contains(name));
        new_toc.push(new_segment.name().await);
        let num_new_toc_segments = new_toc.len();

        let mut new_pending = self.versions.get(&current_version).unwrap().pending.clone();
        let last_sequence_number = self.versions.get(&current_version).unwrap().sequence_number;
        if is_pending {
            new_pending.insert(new_segment.name().await, old_segment_names);
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

        let mut versions_info_write = RwLockUpgradableReadGuard::upgrade(versions_info_read).await;
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
    pub async fn replace_segment_safe(
        &self,
        new_segment: BoxedImmutableSegment<Q>,
        old_segment_names: Vec<String>,
        is_pending: bool,
    ) -> Result<()> {
        let versions_info_read = self.versions_info.upgradable_read().await;

        self.replace_segment(
            new_segment,
            old_segment_names,
            is_pending,
            versions_info_read,
        )
        .await?;
        Ok(())
    }

    /// Returns the number of the current active version.
    pub async fn current_version(&self) -> u64 {
        self.versions_info.read().await.current_version
    }

    /// Returns the number of active references (snapshots) for a specific version.
    pub async fn get_ref_count(&self, version_number: u64) -> usize {
        *self
            .versions_info
            .read()
            .await
            .version_ref_counts
            .get(&version_number)
            .unwrap_or(&0)
    }

    /// Release the ref count for the version once the snapshot is no longer needed.
    pub async fn release_version(&self, version_number: u64) {
        let mut versions_info_write = self.versions_info.write().await;
        let count = *versions_info_write
            .version_ref_counts
            .get(&version_number)
            .unwrap_or(&0);
        versions_info_write
            .version_ref_counts
            .insert(version_number, count - 1);
    }

    /// This is thread-safe, and will increment the ref count for the version.
    async fn get_current_version_and_increment(&self) -> u64 {
        let mut versions_info_write = self.versions_info.write().await;
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

    /// Returns a list of names for all segments currently managed by the collection.
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
    pub async fn get_active_segment_infos(&self) -> SegmentInfoAndVersion {
        let current_version = self.versions_info.read().await.current_version;
        let active_segments = self.versions.get(&current_version).unwrap().toc.clone();
        let mut segment_infos = Vec::new();
        for pair in self
            .all_segments
            .iter()
            .filter(|pair| active_segments.contains(pair.key()))
        {
            segment_infos.push(SegmentInfo {
                name: pair.key().clone(),
                size_in_bytes: pair.value().size_in_bytes_immutable_segments().await,
                num_docs: pair.value().num_docs().await,
            });
        }
        SegmentInfoAndVersion {
            segment_infos,
            version: current_version,
        }
    }

    /// Initializes an optimization task (like merge or vacuum) for a set of segments.
    ///
    /// This creates a `PendingSegment` that wraps the target segments and updates the
    /// collection's version to include this pending segment, allowing it to be
    /// optimized in the background while still being searchable.
    pub async fn init_optimizing(&self, segments: &Vec<String>) -> Result<String> {
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
        )
        .await;
        let new_boxed_segment =
            BoxedImmutableSegment::PendingSegment(Arc::new(RwLock::new(pending_segment)));
        self.replace_segment_safe(new_boxed_segment, segments.clone(), true)
            .await?;

        Ok(random_name)
    }

    /// Finalizes a pending segment by converting it into a persistent immutable segment.
    ///
    /// This involves copying the built index data to a permanent location and
    /// updating the collection's version to replace the pending segment with the finalized one.
    async fn pending_to_finalized(
        &self,
        pending_segment: &str,
        versions_info_read: RwLockUpgradableReadGuard<'_, VersionsInfo>,
    ) -> Result<String> {
        let random_name = format!("segment_{}", rand::random::<u64>());

        // Hardlink the pending segment to the new segment
        let pending_segment_path = format!("{}/{}", self.base_directory, pending_segment);
        let new_segment_path = format!("{}/{}", self.base_directory, random_name);

        // Create and copy content of the pending segment to the new segment
        std::fs::create_dir_all(new_segment_path.clone())?;

        let options = CopyOptions {
            content_only: true,
            ..CopyOptions::default()
        };
        fs_extra::dir::copy(
            pending_segment_path.clone(),
            new_segment_path.clone(),
            &options,
        )?;

        // Replace the pending segment with the new segment
        let index = if let Some(env) = &self.env {
            MultiSpannReader::new(new_segment_path.clone())
                .read_async::<Q>(
                    self.segment_config.posting_list_encoding_type.clone(),
                    self.segment_config.num_features,
                    env.clone(),
                )
                .await?
        } else {
            MultiSpannReader::new(new_segment_path.clone())
                .read::<Q>(
                    self.segment_config.posting_list_encoding_type.clone(),
                    self.segment_config.num_features,
                )
                .await?
        };
        let terms_path = format!("{}/{}/terms", self.base_directory, random_name.clone());
        let new_segment = BoxedImmutableSegment::FinalizedSegment(Arc::new(RwLock::new(
            ImmutableSegment::new(index, random_name.clone(), Some(terms_path)),
        )));
        self.replace_segment(
            new_segment,
            vec![pending_segment.to_string()],
            false,
            versions_info_read,
        )
        .await?;
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
    pub async fn run_optimizer(
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
                let pending_segment = pending_segment.upgradable_read().await;
                temp_storage_dir = pending_segment
                    .temp_invalidated_ids_storage_directory()
                    .await;
                optimizer.optimize(&pending_segment).await?;
                pending_segment.build_index().await?;
                {
                    let mut pending_segment_write =
                        RwLockUpgradableReadGuard::upgrade(pending_segment).await;
                    pending_segment_write.apply_pending_deletions().await?;
                    pending_segment_write.switch_to_internal_index();
                }
            }

            // Remove temporary invalidated ids storage from pending segment.
            std::fs::remove_dir_all(&temp_storage_dir)?;
        }

        // Make the pending segment finalized
        // Note that this function will upgrade toc lock.
        let toc_locked = self.versions_info.upgradable_read().await;
        self.pending_to_finalized(pending_segment, toc_locked).await
    }

    /// Truncates the WAL by removing entries that have already been persisted to immutable segments.
    async fn trim_wal(&self, flushed_seq_no: i64) -> Result<()> {
        if let Some(wal) = &self.wal {
            wal.write().await.trim_wal(flushed_seq_no)?;
        }
        Ok(())
    }

    /// Returns a reference to the map containing all segments managed by this collection.
    pub fn all_segments(&self) -> &DashMap<String, BoxedImmutableSegment<Q>> {
        &self.all_segments
    }

    /// Internal implementation for document removal.
    ///
    /// Performs the invalidation across the mutable segment, pending mutable segment,
    /// and all immutable segments present in the provided TOC.
    #[inline(always)]
    async fn remove_impl(
        version: &TableOfContent,
        all_segments: &DashMap<String, BoxedImmutableSegment<Q>>,
        mutable_segment: &MutableSegment,
        pending_mutable_segment: Option<&PendingMutableSegment>,
        user_id: u128,
        doc_id: u128,
        sequence_number: u64,
    ) -> Result<()> {
        mutable_segment.invalidate(user_id, doc_id, sequence_number)?;

        if let Some(seg) = pending_mutable_segment {
            seg.invalidate(user_id, doc_id)?;
        }

        for segment_name in version.toc.iter() {
            if let Some(segment) = all_segments.get(segment_name) {
                segment.remove(user_id, doc_id).await?;
            }
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
    pub async fn remove(&self, user_id: u128, doc_id: u128, sequence_number: u64) -> Result<()> {
        let version_info_read = self.versions_info.read().await;
        let current_version = version_info_read.current_version;
        let version = self.versions.get(&current_version).unwrap();

        let mutable_segments = self.mutable_segments.read().await;
        let mutable_segment = mutable_segments.mutable_segment.read().await;
        let pending_mutable_segment = mutable_segments.pending_mutable_segment.read().await;

        Self::remove_impl(
            &version,
            &self.all_segments,
            &mutable_segment,
            pending_mutable_segment.as_ref(),
            user_id,
            doc_id,
            sequence_number,
        )
        .await?;

        // The doc is immediately not searchable
        INTERNAL_METRICS.num_searchable_docs_dec(&self.collection_name);

        Ok(())
    }

    /// Iterates through the active segments and initiates vacuuming for segments that exceed the auto-vacuum threshold.
    ///
    /// This function checks each active segment to determine if it should be auto-vacuumed based on its configuration.
    /// If a segment exceeds the threshold, a vacuuming operation is initiated to optimize the segment.
    async fn auto_vacuum(&self) -> Result<()> {
        let segment_infos = self.get_active_segment_infos().await.segment_infos;

        for segment_info in segment_infos.iter() {
            let segment_name = &segment_info.name;
            if let Some(segment) = self.all_segments.get(segment_name) {
                if segment.should_auto_vacuum().await {
                    info!(
                        "{}: Auto vacuuming segment {}",
                        self.collection_name, segment_name
                    );
                    let segments_to_optimize = vec![segment_name.clone()];
                    if let Result::Ok(pending_segment) =
                        self.init_optimizing(&segments_to_optimize).await
                    {
                        let vacuum_optimizer = VacuumOptimizer::<Q>::new();
                        self.run_optimizer(&vacuum_optimizer, &pending_segment)
                            .await?;
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
    async fn auto_merge(&self) -> Result<()> {
        let mut segment_infos = self.get_active_segment_infos().await.segment_infos;
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

        if let Result::Ok(pending_segment) = self.init_optimizing(&segmens_to_merge).await {
            let merge_optimizer = MergeOptimizer::<Q>::new();
            self.run_optimizer(&merge_optimizer, &pending_segment)
                .await?;
        } else {
            warn!(
                "{}: Failed to init optimizing segment {:?}. Skipping",
                self.collection_name, segmens_to_merge
            );
        }

        Ok(())
    }

    /// Checks for segments that should be auto-vacuumed and initiates optimization for them.
    pub async fn auto_optimize(&self) -> Result<()> {
        debug!("{}: Auto optimizing", self.collection_name);
        self.auto_vacuum().await?;
        self.auto_merge().await?;

        Ok(())
    }
}

// Test
#[cfg(test)]
mod tests {

    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;
    use std::time::Duration;

    use anyhow::{Ok, Result};
    use async_lock::RwLock;
    use config::attribute_schema::{AttributeSchema, AttributeType, Language};
    use config::collection::CollectionConfig;
    use config::search_params::SearchParams;
    use metrics::INTERNAL_METRICS;
    use proto::muopdb::DocumentAttribute;
    use quantization::noq::noq::NoQuantizerL2;
    use rand::Rng;
    use tempdir::TempDir;
    use tokio::task::JoinSet;

    use crate::collection::core::Collection;
    use crate::collection::reader::CollectionReader;
    use crate::collection::snapshot::Snapshot;
    use crate::optimizers::noop::NoopOptimizer;
    use crate::segment::{BoxedImmutableSegment, MockedSegment, Segment};
    use crate::wal::entry::WalOpType;

    // Used for multi-threaded WAL write tests
    const USER_ID: u128 = 0;
    const NUM_GROUPS: usize = 10; // 10 groups of 4 operations each
    const OPERATIONS_PER_GROUP: usize = 4; // 3 inserts + 1 delete per group

    #[tokio::test]
    async fn test_collection() -> Result<()> {
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
                .await
                .unwrap();
        }
        let current_version = collection.current_version().await;
        assert_eq!(current_version, 1);

        let version_1 = 1;
        {
            let snapshot = collection.clone().get_snapshot().await?;
            assert_eq!(snapshot.segments.len(), 2);

            let ref_count = collection.clone().get_ref_count(version_1).await;
            assert_eq!(ref_count, 1);
        }

        // Snapshot should be dropped when it goes out of scope
        // Snapshot should be dropped when it goes out of scope
        // Since Snapshot::drop spawns a background task, we need to wait for it
        let mut ref_count = collection.clone().get_ref_count(version_1).await;
        for _ in 0..100 {
            if ref_count == 0 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
            ref_count = collection.clone().get_ref_count(version_1).await;
        }
        assert_eq!(ref_count, 0);

        // Create another snapshot, then add new segments
        let version_2 = 2;
        {
            let snapshot = collection.clone().get_snapshot().await?;
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
                .await
                .unwrap();

            let ref_count = collection.clone().get_ref_count(version_1).await;
            assert_eq!(ref_count, 1);

            let ref_count = collection.clone().get_ref_count(version_2).await;
            assert_eq!(ref_count, 0);
        }

        // Snapshot should be dropped when it goes out of scope
        // Snapshot should be dropped when it goes out of scope
        // Since Snapshot::drop spawns a background task, we need to wait for it
        let mut ref_count = collection.clone().get_ref_count(version_1).await;
        for _ in 0..100 {
            if ref_count == 0 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
            ref_count = collection.clone().get_ref_count(version_1).await;
        }
        assert_eq!(ref_count, 0);
        let ref_count = collection.clone().get_ref_count(version_2).await;
        assert_eq!(ref_count, 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_collection_multi_thread() -> Result<()> {
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
        tokio::spawn(async move {
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
                .await
                .unwrap();

            while !stopped_cpy.load(std::sync::atomic::Ordering::Relaxed) {
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            }
        });

        // Sleep until there is a new version
        let mut latest_version = collection.clone().current_version().await;
        while latest_version != 1 {
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            latest_version = collection.clone().current_version().await;
        }

        // Create another thread to get a snapshot
        let collection_cpy = collection.clone();
        let stopped_cpy = stopped.clone();
        tokio::spawn(async move {
            let snapshot = collection_cpy.clone().get_snapshot().await.unwrap();
            assert_eq!(snapshot.segments.len(), 2);
            assert_eq!(snapshot.version(), 1);

            let version_1 = 1;
            let ref_count = collection_cpy.clone().get_ref_count(version_1).await;
            assert_eq!(ref_count, 1);

            while !stopped_cpy.load(std::sync::atomic::Ordering::Relaxed) {
                assert_eq!(snapshot.version(), 1);
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            }
        });

        // Sleep for 200ms, then check ref count
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        let version_1 = 1;
        let ref_count = collection.clone().get_ref_count(version_1).await;
        assert_eq!(ref_count, 1);

        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
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
        collection
            .insert_for_users(
                &[0],
                1,
                &[1.0, 2.0, 3.0, 4.0],
                0,
                DocumentAttribute::default(),
            )
            .await?;
        collection.flush().await?;

        let segment_names = collection.get_all_segment_names();
        assert_eq!(segment_names.len(), 1);

        let pending_segment = collection.init_optimizing(&segment_names).await?;

        let snapshot = collection.clone().get_snapshot().await?;
        let snapshot = Arc::new(snapshot);
        assert_eq!(snapshot.segments.len(), 1);
        assert_eq!(snapshot.version(), 2);
        let segment_name = snapshot.segments[0].name().await;
        assert_eq!(segment_name, pending_segment);

        let result = Snapshot::search_for_users(
            snapshot,
            &[0],
            vec![1.0, 2.0, 3.0, 4.0],
            &SearchParams::new(10, 10, false),
            None,
        )
        .await
        .unwrap();
        assert_eq!(result.id_with_scores.len(), 1);
        assert_eq!(result.id_with_scores[0].doc_id, 1);

        let optimizer = NoopOptimizer::new();
        collection
            .run_optimizer(&optimizer, &pending_segment)
            .await?;

        let snapshot = collection.clone().get_snapshot().await?;
        assert_eq!(snapshot.segments.len(), 1);
        assert_eq!(snapshot.version(), 3);

        let toc = collection.get_current_toc().await;
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

        collection
            .insert_for_users(
                &[0],
                1,
                &[1.0, 2.0, 3.0, 4.0],
                0,
                DocumentAttribute::default(),
            )
            .await?;
        collection.flush().await?;

        // A thread to optimize the segment
        let collection_cpy_for_optimizer = collection.clone();
        let collection_cpy_for_query = collection.clone();

        let stopped = Arc::new(AtomicBool::new(false));
        let stopped_cpy_for_optimizer = stopped.clone();
        tokio::spawn(async move {
            while !stopped_cpy_for_optimizer.load(std::sync::atomic::Ordering::Relaxed) {
                let c = collection_cpy_for_optimizer.clone();
                let snapshot = c.get_snapshot().await.unwrap();
                let mut segment_names = vec![];
                for s in snapshot.segments.iter() {
                    segment_names.push(s.name().await);
                }
                let pending_segment = collection_cpy_for_optimizer
                    .init_optimizing(&segment_names)
                    .await
                    .unwrap();

                let toc = collection_cpy_for_optimizer.get_current_toc().await;
                assert_eq!(toc.pending.len(), 1);

                // Sleep randomly between 100ms and 200ms
                let sleep_duration = rand::thread_rng().gen_range(100..200);
                tokio::time::sleep(std::time::Duration::from_millis(sleep_duration)).await;

                let optimizer = NoopOptimizer::new();
                collection_cpy_for_optimizer
                    .run_optimizer(&optimizer, &pending_segment)
                    .await
                    .unwrap();

                let toc = collection_cpy_for_optimizer.get_current_toc().await;
                assert_eq!(toc.pending.len(), 0);
            }
        });

        // A thread to query the collection
        let stopped_cpy_for_query = stopped.clone();
        tokio::spawn(async move {
            while !stopped_cpy_for_query.load(std::sync::atomic::Ordering::Relaxed) {
                let c = collection_cpy_for_query.clone();
                let snapshot = c.get_snapshot().await.unwrap();
                let snapshot = Arc::new(snapshot);
                let result = Snapshot::search_for_users(
                    snapshot,
                    &[0],
                    vec![1.0, 2.0, 3.0, 4.0],
                    &SearchParams::new(10, 10, false),
                    None,
                )
                .await
                .unwrap();

                assert_eq!(result.id_with_scores.len(), 1);
                assert_eq!(result.id_with_scores[0].doc_id, 1);

                // Sleep randomly between 100ms and 200ms
                let sleep_duration = rand::thread_rng().gen_range(100..200);
                tokio::time::sleep(std::time::Duration::from_millis(sleep_duration)).await;
            }
        });

        // Sleep for 5 seconds, then stop the threads
        tokio::time::sleep(std::time::Duration::from_millis(5000)).await;
        stopped.store(true, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    #[tokio::test]
    async fn test_collection_reader() -> Result<()> {
        let collection_name = "test_collection";
        let temp_dir = TempDir::new(collection_name)?;
        let base_directory: String = temp_dir.path().to_str().unwrap().to_string();
        let segment_config = CollectionConfig::default_test_config();
        // write the collection config
        let collection_config_path = format!("{base_directory}/collection_config.json");
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

            collection
                .insert_for_users(
                    &[0],
                    1,
                    &[1.0, 2.0, 3.0, 4.0],
                    0,
                    DocumentAttribute::default(),
                )
                .await?;
            collection.flush().await?;

            let segment_names = collection.get_all_segment_names();
            assert_eq!(segment_names.len(), 1);

            let pending_segment = collection.init_optimizing(&segment_names).await?;

            let toc = collection.get_current_toc().await;
            assert_eq!(toc.pending.len(), 1);
            assert_eq!(toc.pending.get(&pending_segment).unwrap().len(), 1);
        }

        let reader = CollectionReader::new(collection_name.to_string(), base_directory, None);
        let collection = reader.read::<NoQuantizerL2>().await?;
        let toc = collection.get_current_toc().await;
        assert_eq!(toc.pending.len(), 1);
        Ok(())
    }

    #[tokio::test]
    async fn test_collection_with_wal() -> Result<()> {
        env_logger::try_init().ok();
        let collection_name = "test_collection";
        let temp_dir = TempDir::new(collection_name)?;
        let base_directory: String = temp_dir.path().to_str().unwrap().to_string();
        let segment_config = CollectionConfig {
            wal_file_size: 1024 * 1024,
            ..CollectionConfig::default_test_config()
        };

        let collection = Arc::new(
            Collection::<NoQuantizerL2>::new(
                collection_name.to_string(),
                base_directory.clone(),
                segment_config,
            )
            .unwrap(),
        );

        collection
            .write_to_wal(
                Arc::from([1u128]),
                Arc::from([0u128]),
                WalOpType::Insert(Arc::from([1.0, 2.0, 3.0, 4.0])),
                None,
            )
            .await?;
        collection
            .write_to_wal(
                Arc::from([2u128]),
                Arc::from([0u128]),
                WalOpType::Insert(Arc::from([1.0, 2.0, 3.0, 4.0])),
                None,
            )
            .await?;
        collection
            .write_to_wal(
                Arc::from([3u128]),
                Arc::from([0u128]),
                WalOpType::Insert(Arc::from([1.0, 2.0, 3.0, 4.0])),
                None,
            )
            .await?;
        collection
            .write_to_wal(
                Arc::from([4u128]),
                Arc::from([0u128]),
                WalOpType::Insert(Arc::from([1.0, 2.0, 3.0, 4.0])),
                None,
            )
            .await?;
        collection
            .write_to_wal(
                Arc::from([5u128]),
                Arc::from([0u128]),
                WalOpType::Insert(Arc::from([1.0, 2.0, 3.0, 4.0])),
                None,
            )
            .await?;
        collection
            .write_to_wal(
                Arc::from([5u128]),
                Arc::from([0u128]),
                WalOpType::Delete,
                None,
            )
            .await?;

        // Process all ops
        let mut ops_processed = 0;
        loop {
            let op = collection.process_one_op().await?;
            if op == 0 {
                break;
            }
            ops_processed += 1;
        }
        eprintln!("Processed {} ops", ops_processed);
        collection.flush().await?;
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
                assert!(immutable_segment
                    .read()
                    .await
                    .get_point_id(0, 1)
                    .await
                    .is_some());
                assert!(immutable_segment
                    .read()
                    .await
                    .get_point_id(0, 2)
                    .await
                    .is_some());
                assert!(immutable_segment
                    .read()
                    .await
                    .get_point_id(0, 3)
                    .await
                    .is_some());
                assert!(immutable_segment
                    .read()
                    .await
                    .get_point_id(0, 4)
                    .await
                    .is_some());
                assert!(immutable_segment
                    .read()
                    .await
                    .get_point_id(0, 5)
                    .await
                    .is_none());
            }
            _ => {
                panic!("Expected FinalizedSegment");
            }
        }
        let toc = collection.get_current_toc().await;
        assert_eq!(toc.pending.len(), 0);
        assert_eq!(toc.sequence_number, 5);

        Ok(())
    }

    #[tokio::test]
    async fn test_collection_hybrid_search() -> Result<()> {
        use std::collections::HashMap;

        let collection_name = "test_collection_hybrid_search";
        let temp_dir = TempDir::new(collection_name)?;
        let base_directory: String = temp_dir.path().to_str().unwrap().to_string();

        let mut segment_config = CollectionConfig::default_test_config();
        segment_config.num_features = 4;

        let mut schema_map = HashMap::new();
        schema_map.insert("title".to_string(), AttributeType::Text(Language::English));
        schema_map.insert("category".to_string(), AttributeType::Keyword);
        segment_config.attribute_schema = Some(AttributeSchema::new(schema_map));

        let collection = Arc::new(
            Collection::<NoQuantizerL2>::new(
                collection_name.to_string(),
                base_directory.clone(),
                segment_config,
            )
            .unwrap(),
        );

        let mut attributes1 = HashMap::new();
        attributes1.insert(
            "title".to_string(),
            proto::muopdb::AttributeValue {
                value: Some(proto::muopdb::attribute_value::Value::TextValue(
                    "apple banana cherry".to_string(),
                )),
            },
        );
        attributes1.insert(
            "category".to_string(),
            proto::muopdb::AttributeValue {
                value: Some(proto::muopdb::attribute_value::Value::KeywordValue(
                    "fruit".to_string(),
                )),
            },
        );
        let doc_attr1 = proto::muopdb::DocumentAttribute { value: attributes1 };

        let mut attributes2 = HashMap::new();
        attributes2.insert(
            "title".to_string(),
            proto::muopdb::AttributeValue {
                value: Some(proto::muopdb::attribute_value::Value::TextValue(
                    "banana orange".to_string(),
                )),
            },
        );
        attributes2.insert(
            "category".to_string(),
            proto::muopdb::AttributeValue {
                value: Some(proto::muopdb::attribute_value::Value::KeywordValue(
                    "citrus".to_string(),
                )),
            },
        );
        let doc_attr2 = proto::muopdb::DocumentAttribute { value: attributes2 };

        let mut attributes3 = HashMap::new();
        attributes3.insert(
            "title".to_string(),
            proto::muopdb::AttributeValue {
                value: Some(proto::muopdb::attribute_value::Value::TextValue(
                    "dog cat mouse".to_string(),
                )),
            },
        );
        attributes3.insert(
            "category".to_string(),
            proto::muopdb::AttributeValue {
                value: Some(proto::muopdb::attribute_value::Value::KeywordValue(
                    "animal".to_string(),
                )),
            },
        );
        let doc_attr3 = proto::muopdb::DocumentAttribute { value: attributes3 };

        collection
            .insert_for_users(&[0], 1, &[1.0, 2.0, 3.0, 4.0], 0, doc_attr1)
            .await?;
        collection
            .insert_for_users(&[0], 2, &[5.0, 6.0, 7.0, 8.0], 1, doc_attr2)
            .await?;
        collection
            .insert_for_users(&[0], 3, &[9.0, 10.0, 11.0, 12.0], 2, doc_attr3)
            .await?;

        collection.flush().await?;

        let snapshot = collection.clone().get_snapshot().await?;
        let snapshot = Arc::new(snapshot);

        let query = vec![5.0, 6.0, 7.0, 8.0];
        let k = 10;
        let ef_construction = 10;
        let record_pages = false;

        let params = SearchParams::new(k, ef_construction, record_pages);
        let result =
            Snapshot::search_for_users(snapshot.clone(), &[0], query.clone(), &params, None)
                .await
                .unwrap();

        assert_eq!(result.id_with_scores.len(), 3);

        let contains_filter = proto::muopdb::ContainsFilter {
            path: "title".to_string(),
            value: "apple".to_string(),
        };
        let document_filter = proto::muopdb::DocumentFilter {
            filter: Some(proto::muopdb::document_filter::Filter::Contains(
                contains_filter,
            )),
        };

        let result = Snapshot::search_for_users(
            snapshot.clone(),
            &[0],
            query.clone(),
            &params,
            Some(Arc::new(document_filter)),
        )
        .await
        .unwrap();

        assert_eq!(
            result.id_with_scores.len(),
            1,
            "Expected only doc 1 to have 'apple' in title, got {:?}",
            result
                .id_with_scores
                .iter()
                .map(|r| r.doc_id)
                .collect::<Vec<_>>()
        );
        assert_eq!(result.id_with_scores[0].doc_id, 1);

        let contains_filter = proto::muopdb::ContainsFilter {
            path: "title".to_string(),
            value: "orange".to_string(),
        };
        let document_filter = proto::muopdb::DocumentFilter {
            filter: Some(proto::muopdb::document_filter::Filter::Contains(
                contains_filter,
            )),
        };

        let result = Snapshot::search_for_users(
            snapshot.clone(),
            &[0],
            query.clone(),
            &params,
            Some(Arc::new(document_filter)),
        )
        .await
        .unwrap();

        assert_eq!(
            result.id_with_scores.len(),
            1,
            "Expected only doc 2 to have 'orange' in title, got {:?}",
            result
                .id_with_scores
                .iter()
                .map(|r| r.doc_id)
                .collect::<Vec<_>>()
        );
        assert_eq!(result.id_with_scores[0].doc_id, 2);

        let contains_filter = proto::muopdb::ContainsFilter {
            path: "title".to_string(),
            value: "dog".to_string(),
        };
        let document_filter = proto::muopdb::DocumentFilter {
            filter: Some(proto::muopdb::document_filter::Filter::Contains(
                contains_filter,
            )),
        };

        let result = Snapshot::search_for_users(
            snapshot,
            &[0],
            query.clone(),
            &params,
            Some(Arc::new(document_filter)),
        )
        .await
        .unwrap();

        assert_eq!(
            result.id_with_scores.len(),
            1,
            "Expected only doc 3 to have 'dog' in title, got {:?}",
            result
                .id_with_scores
                .iter()
                .map(|r| r.doc_id)
                .collect::<Vec<_>>()
        );
        assert_eq!(result.id_with_scores[0].doc_id, 3);

        Ok(())
    }

    #[tokio::test]
    async fn test_collection_with_wal_reopen() -> Result<()> {
        env_logger::try_init().ok();
        let collection_name = "test_collection";
        let temp_dir = TempDir::new(collection_name)?;
        let base_directory: String = temp_dir.path().to_str().unwrap().to_string();
        let segment_config = CollectionConfig {
            wal_file_size: 1024 * 1024,
            ..CollectionConfig::default_test_config()
        };
        Collection::<NoQuantizerL2>::init_new_collection(base_directory.clone(), &segment_config)?;

        // Insert but don't flush
        {
            let reader =
                CollectionReader::new(collection_name.to_string(), base_directory.clone(), None);
            let collection = reader.read::<NoQuantizerL2>().await?;

            collection
                .write_to_wal(
                    Arc::from([1u128]),
                    Arc::from([0u128]),
                    WalOpType::Insert(Arc::from([1.0, 2.0, 3.0, 4.0])),
                    None,
                )
                .await?;
            collection
                .write_to_wal(
                    Arc::from([2u128]),
                    Arc::from([0u128]),
                    WalOpType::Insert(Arc::from([1.0, 2.0, 3.0, 4.0])),
                    None,
                )
                .await?;
            collection
                .write_to_wal(
                    Arc::from([1u128]),
                    Arc::from([0u128]),
                    WalOpType::Delete,
                    None,
                )
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
            let reader =
                CollectionReader::new(collection_name.to_string(), base_directory.clone(), None);
            let collection = reader.read::<NoQuantizerL2>().await?;

            let toc = collection.get_current_toc().await;
            assert_eq!(toc.pending.len(), 0);
            assert_eq!(toc.sequence_number, -1);

            collection.flush().await?;
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
                    assert!(immutable_segment
                        .read()
                        .await
                        .get_point_id(0, 1)
                        .await
                        .is_none());
                    assert!(immutable_segment
                        .read()
                        .await
                        .get_point_id(0, 2)
                        .await
                        .is_some());
                }
                _ => {
                    panic!("Expected FinalizedSegment");
                }
            }
            let toc = collection.get_current_toc().await;
            assert_eq!(toc.pending.len(), 0);
            assert_eq!(toc.sequence_number, 2);

            // Write 2 more ops, but don't flush
            collection
                .write_to_wal(
                    Arc::from([2u128]),
                    Arc::from([0u128]),
                    WalOpType::Insert(Arc::from([1.0, 2.0, 3.0, 4.0])),
                    None,
                )
                .await?;
            collection
                .write_to_wal(
                    Arc::from([3u128]),
                    Arc::from([0u128]),
                    WalOpType::Insert(Arc::from([1.0, 2.0, 3.0, 4.0])),
                    None,
                )
                .await?;
            collection
                .write_to_wal(
                    Arc::from([4u128]),
                    Arc::from([0u128]),
                    WalOpType::Insert(Arc::from([1.0, 2.0, 3.0, 4.0])),
                    None,
                )
                .await?;
            collection
                .write_to_wal(
                    Arc::from([1u128]),
                    Arc::from([0u128]),
                    WalOpType::Delete,
                    None,
                )
                .await?;
            collection
                .write_to_wal(
                    Arc::from([2u128]),
                    Arc::from([0u128]),
                    WalOpType::Delete,
                    None,
                )
                .await?;
            collection
                .write_to_wal(
                    Arc::from([3u128]),
                    Arc::from([0u128]),
                    WalOpType::Delete,
                    None,
                )
                .await?;
        }

        {
            let reader = CollectionReader::new(collection_name.to_string(), base_directory, None);
            let collection = reader.read::<NoQuantizerL2>().await?;

            let toc = collection.get_current_toc().await;
            assert_eq!(toc.pending.len(), 0);
            assert_eq!(toc.sequence_number, 2);

            collection.flush().await?;
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
                    assert!(immutable_segment
                        .read()
                        .await
                        .get_point_id(0, 1)
                        .await
                        .is_none());
                    assert!(immutable_segment
                        .read()
                        .await
                        .get_point_id(0, 2)
                        .await
                        .is_none());
                    assert!(immutable_segment
                        .read()
                        .await
                        .get_point_id(0, 3)
                        .await
                        .is_none());
                }
                _ => {
                    panic!("Expected FinalizedSegment");
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
                    assert!(immutable_segment.read().await.is_invalidated(0, 2).await?);
                }
                _ => {
                    panic!("Expected FinalizedSegment");
                }
            }
            let toc = collection.get_current_toc().await;
            assert_eq!(toc.pending.len(), 0);
            assert_eq!(toc.sequence_number, 8);
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_collection_inval() {
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
            .insert_for_users(
                &[0],
                1,
                &[1.0, 2.0, 3.0, 4.0],
                0,
                DocumentAttribute::default()
            )
            .await
            .is_ok());
        assert!(collection
            .insert_for_users(
                &[0],
                2,
                &[2.0, 2.0, 3.0, 4.0],
                1,
                DocumentAttribute::default()
            )
            .await
            .is_ok());
        assert!(collection
            .insert_for_users(
                &[0],
                3,
                &[3.0, 2.0, 3.0, 4.0],
                2,
                DocumentAttribute::default()
            )
            .await
            .is_ok());
        assert!(collection.remove(0, 2, 3).await.is_ok());

        assert!(collection.flush().await.is_ok());
        assert!(collection
            .insert_for_users(
                &[0],
                1,
                &[1.0, 2.0, 3.0, 4.0],
                0,
                DocumentAttribute::default()
            )
            .await
            .is_ok());
        assert!(collection
            .insert_for_users(
                &[0],
                2,
                &[1.0, 2.0, 3.0, 4.0],
                1,
                DocumentAttribute::default()
            )
            .await
            .is_ok());
        assert!(collection
            .insert_for_users(
                &[0],
                3,
                &[2.0, 2.0, 3.0, 4.0],
                2,
                DocumentAttribute::default()
            )
            .await
            .is_ok());
        assert!(collection
            .insert_for_users(
                &[0],
                4,
                &[3.0, 2.0, 3.0, 4.0],
                3,
                DocumentAttribute::default()
            )
            .await
            .is_ok());
        assert!(collection.remove(0, 2, 4).await.is_ok());
        assert!(collection.remove(0, 3, 5).await.is_ok());
        assert!(collection.remove(0, 4, 6).await.is_ok());

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
                assert!(immutable_segment
                    .read()
                    .await
                    .get_point_id(0, 1)
                    .await
                    .is_some());
                assert!(immutable_segment
                    .read()
                    .await
                    .get_point_id(0, 2)
                    .await
                    .is_none());
                assert!(immutable_segment
                    .read()
                    .await
                    .get_point_id(0, 3)
                    .await
                    .is_some());
                assert!(immutable_segment
                    .read()
                    .await
                    .is_invalidated(0, 3)
                    .await
                    .unwrap());

                {
                    let mutable_segments_guard = collection.mutable_segments.read().await;
                    let mutable_segment_guard = mutable_segments_guard.mutable_segment.read().await;

                    assert!(mutable_segment_guard.is_valid_doc_id(0, 1));
                    assert!(!mutable_segment_guard.is_valid_doc_id(0, 2));
                    assert!(!mutable_segment_guard.is_valid_doc_id(0, 3));
                    assert!(!mutable_segment_guard.is_valid_doc_id(0, 4));
                }
            }
            _ => {
                panic!("Expected FinalizedSegment");
            }
        }
    }

    /// Tests invalidation logic with a larger number of documents.
    ///
    /// This test inserts 20 documents into the collection, then removes 5 of them.
    /// It verifies that the removed documents are marked as invalid in the mutable segment
    /// and that the remaining documents are still valid.
    #[tokio::test]
    async fn test_collection_inval_large() {
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
                .insert_for_users(
                    &[user_id],
                    i as u128,
                    &vector,
                    i as u64,
                    DocumentAttribute::default()
                )
                .await
                .is_ok());
        }

        // Delete 5 data points (e.g., doc_ids 0 to 4)
        for i in 0..num_docs_to_delete {
            assert!(collection
                .remove(user_id, i as u128, (num_docs_to_add + i) as u64)
                .await
                .is_ok());
        }

        // Verify that the 5 deleted documents are invalidated
        for i in 0..num_docs_to_delete {
            let mutable_segments_guard = collection.mutable_segments.read().await;
            let mutable_segment_guard = mutable_segments_guard.mutable_segment.read().await;
            assert!(!mutable_segment_guard.is_valid_doc_id(user_id, i as u128));
        }

        // Verify that the remaining 15 documents are still valid
        for i in num_docs_to_delete..num_docs_to_add {
            let mutable_segments_guard = collection.mutable_segments.read().await;
            let mutable_segment_guard = mutable_segments_guard.mutable_segment.read().await;
            assert!(mutable_segment_guard.is_valid_doc_id(user_id, i as u128));
        }
    }

    #[tokio::test]
    async fn test_collection_metrics() {
        let collection_name = "test_collection_metrics";
        let temp_dir = TempDir::new(collection_name).unwrap();
        let base_directory: String = temp_dir.path().to_str().unwrap().to_string();
        let segment_config = CollectionConfig {
            num_features: 16, // Set a specific feature size for precise calculations
            ..CollectionConfig::default()
        };
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
                .insert_for_users(&user_ids, doc_id, vector, 0, DocumentAttribute::default())
                .await
                .unwrap();
        }

        // Verify metrics after insertions
        assert_eq!(get_searchable_docs(), 5);
        assert_eq!(get_active_segments(), 0);

        // Flush the mutable segment
        println!("Flushing the collection once...");
        assert!(collection.flush().await.is_ok());
        {
            let mutable_segments_guard = collection.mutable_segments.read().await;
            let mutable_segment_guard = mutable_segments_guard.mutable_segment.read().await;
            assert_eq!(mutable_segment_guard.num_docs(), 0);
        }

        // Verify metrics after flush
        assert_eq!(get_searchable_docs(), 5);
        assert_eq!(get_active_segments(), 1);

        // Remove 2 documents for first user
        println!("Removing 2 documents...");
        let docs_to_delete = &doc_ids[0..2];
        for &doc_id in docs_to_delete {
            collection.remove(user_ids[0], doc_id, 0).await.unwrap();
        }

        // Verify metrics after document removal
        assert_eq!(get_searchable_docs(), 3);
        assert_eq!(get_active_segments(), 1);

        // Flush again. Should not create a new segment
        println!("Flushing the collection again...");
        assert!(collection.flush().await.is_ok());
        assert_eq!(get_active_segments(), 1);

        // Insert 2 documents for all users
        println!("Inserting the 2 documents back...");
        let docs_to_insert = docs_to_delete;
        for (idx, &doc_id) in docs_to_insert.iter().enumerate() {
            let vector = &test_vectors[idx * num_features..(idx + 1) * num_features];
            collection
                .insert_for_users(&user_ids, doc_id, vector, 0, DocumentAttribute::default())
                .await
                .unwrap();
        }

        // Flush again to create another immutable segment
        println!("Flushing the collection again...");
        assert!(collection.flush().await.is_ok());

        // Final metrics verification
        assert_eq!(get_searchable_docs(), 5);
        assert_eq!(get_active_segments(), 2);
    }

    /// Tests concurrent WAL write groups with mixed insert and delete operations
    ///
    /// This test spawns multiple concurrent async tasks that write both inserts and deletes
    /// to the WAL simultaneously. Each group contains 3 inserts and 1 delete,
    /// testing the write group mechanism under real concurrent load with mixed operations.
    async fn test_collection_concurrent_wal_write_groups_with_size(
        wal_write_group_size: usize,
    ) -> Result<()> {
        let collection_name = &format!("test_collection_concurrent_wal_{wal_write_group_size}");
        let temp_dir = TempDir::new(collection_name)?;
        let base_directory: String = temp_dir.path().to_str().unwrap().to_string();
        let segment_config = CollectionConfig {
            wal_file_size: 1024 * 1024,
            wal_write_group_size,
            ..CollectionConfig::default_test_config()
        };
        let collection = Arc::new(
            Collection::<NoQuantizerL2>::new(
                collection_name.to_string(),
                base_directory.clone(),
                segment_config,
            )
            .unwrap(),
        );

        let mut join_set = JoinSet::new();

        // Spawn multiple concurrent async tasks organized in groups
        for group_idx in 0..NUM_GROUPS {
            for op_idx in 0..OPERATIONS_PER_GROUP {
                let collection_clone = collection.clone();
                let task_id = group_idx * OPERATIONS_PER_GROUP + op_idx;

                join_set.spawn(async move {
                    if op_idx == OPERATIONS_PER_GROUP - 1 {
                        // Last operation in each group is a delete
                        // Delete the document from the first insert in this group
                        let doc_to_delete = (group_idx * OPERATIONS_PER_GROUP) as u128;

                        collection_clone
                            .write_to_wal(
                                Arc::from([doc_to_delete]),
                                Arc::from([USER_ID]),
                                WalOpType::Delete,
                                None,
                            )
                            .await
                            .unwrap();
                    } else {
                        // First 3 operations in each group are inserts
                        let doc_id = task_id as u128;
                        let vector_data = vec![
                            doc_id as f32,
                            (doc_id + 1) as f32,
                            (doc_id + 2) as f32,
                            (doc_id + 3) as f32,
                        ];

                        collection_clone
                            .write_to_wal(
                                Arc::from([doc_id]),
                                Arc::from([USER_ID]),
                                WalOpType::Insert(Arc::from(vector_data.as_slice())),
                                None,
                            )
                            .await
                            .unwrap();
                    }
                });
            }
        }

        // Wait for all tasks to complete concurrently
        while let Some(result) = join_set.join_next().await {
            result.unwrap(); // Ensure no task panicked
        }

        // Process all ops from the WAL
        loop {
            let ops_processed = collection.process_one_op().await?;
            if ops_processed == 0 {
                break;
            }
        }

        // Flush and verify results
        collection.flush().await?;
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
                // Verify the results for each group
                for group_idx in 0..NUM_GROUPS {
                    let base_doc_id = (group_idx * OPERATIONS_PER_GROUP) as u128;

                    // First document in each group should be deleted (not exist)
                    let deleted_doc_id = base_doc_id;
                    assert!(
                        immutable_segment
                            .read()
                            .await
                            .get_point_id(0, deleted_doc_id)
                            .await
                            .is_none(),
                        "Document {deleted_doc_id} should be deleted",
                    );

                    // Other documents in the group should exist (inserted but not deleted)
                    for op_idx in 1..OPERATIONS_PER_GROUP - 1 {
                        // The 2 middle documents in each group
                        let existing_doc_id = base_doc_id + op_idx as u128;
                        assert!(
                            immutable_segment
                                .read()
                                .await
                                .get_point_id(0, existing_doc_id)
                                .await
                                .is_some(),
                            "Document {existing_doc_id} should exist",
                        );
                    }
                }
            }
            _ => {
                panic!("Expected FinalizedSegment");
            }
        }

        let toc = collection.get_current_toc().await;
        assert_eq!(toc.pending.len(), 0);

        // Should have sequence number equal to total number of operations - 1
        let total_operations = NUM_GROUPS * OPERATIONS_PER_GROUP;
        assert_eq!(toc.sequence_number, (total_operations - 1) as i64);

        Ok(())
    }

    /// Original test function that calls the parameterized version with different group sizes
    #[tokio::test]
    async fn test_collection_concurrent_wal_write_groups() -> Result<()> {
        env_logger::try_init().ok();

        // Group size = 0 : every write is its own leader
        test_collection_concurrent_wal_write_groups_with_size(0).await?;

        // Group size < number of writes - 1 : multiple groups with leaders without timeout, except
        // possibly the last one
        // Here we set group size to <= 1/3 of total writes to ensure multiple groups
        test_collection_concurrent_wal_write_groups_with_size(
            (NUM_GROUPS * OPERATIONS_PER_GROUP) / 3,
        )
        .await?;

        // Group size = number of writes - 1 : last write is the leader
        test_collection_concurrent_wal_write_groups_with_size(
            NUM_GROUPS * OPERATIONS_PER_GROUP - 1,
        )
        .await?;

        // Group size = number of writes : last write takes over and becomes leader after timeout
        test_collection_concurrent_wal_write_groups_with_size(NUM_GROUPS * OPERATIONS_PER_GROUP)
            .await?;

        Ok(())
    }

    /// Test focused on timeout behavior with slow operations
    #[tokio::test]
    async fn test_collection_wal_write_group_timeout_behavior() -> Result<()> {
        env_logger::try_init().ok();

        let collection_name = "test_collection_wal_timeout";
        let temp_dir = TempDir::new(collection_name)?;
        let base_directory: String = temp_dir.path().to_str().unwrap().to_string();
        let segment_config = CollectionConfig {
            wal_file_size: 1024 * 1024,
            // Large group size to have followers per group (> 0 is enough)
            wal_write_group_size: 1000,
            ..CollectionConfig::default_test_config()
        };

        let collection = Arc::new(
            Collection::<NoQuantizerL2>::new(
                collection_name.to_string(),
                base_directory.clone(),
                segment_config,
            )
            .unwrap(),
        );

        let mut join_set = JoinSet::new();

        // Spawn 3 concurrent tasks with delays to test timeout
        for i in 0..3 {
            let collection_clone = collection.clone();
            join_set.spawn(async move {
                // Add delay to ensure we hit timeout rather than group size limit
                tokio::time::sleep(std::time::Duration::from_millis(i * 1000)).await;

                let doc_id = i as u128;
                let vector_data = vec![
                    doc_id as f32,
                    (doc_id + 1) as f32,
                    (doc_id + 2) as f32,
                    (doc_id + 3) as f32,
                ];

                collection_clone
                    .write_to_wal(
                        Arc::from([doc_id]),
                        Arc::from([USER_ID]),
                        WalOpType::Insert(Arc::from(vector_data.as_slice())),
                        None,
                    )
                    .await
                    .unwrap();
            });
        }

        // Wait for all tasks to complete
        while let Some(result) = join_set.join_next().await {
            result.unwrap();
        }

        // Process all ops from the WAL
        loop {
            let ops_processed = collection.process_one_op().await?;
            if ops_processed == 0 {
                break;
            }
        }

        // Flush and verify results
        collection.flush().await?;
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
                // All 3 documents should exist
                for i in 0..3 {
                    assert!(
                        immutable_segment
                            .read()
                            .await
                            .get_point_id(0, i as u128)
                            .await
                            .is_some(),
                        "Document {i} should exist",
                    );
                }
            }
            _ => {
                panic!("Expected FinalizedSegment");
            }
        }

        let toc = collection.get_current_toc().await;
        assert_eq!(toc.pending.len(), 0);
        assert_eq!(toc.sequence_number, 2); // 3 operations, 0-indexed

        Ok(())
    }
}
