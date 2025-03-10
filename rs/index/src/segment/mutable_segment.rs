use std::collections::HashMap;
use std::sync::atomic::AtomicU64;
use std::time::Instant;

use anyhow::{Ok, Result};
use config::collection::CollectionConfig;
use log::debug;
use parking_lot::RwLock;

use crate::ivf::files::invalidated_ids::InvalidatedIdsStorage;
use crate::multi_spann::builder::MultiSpannBuilder;
use crate::multi_spann::writer::MultiSpannWriter;

pub struct MutableSegment {
    multi_spann_builder: MultiSpannBuilder,

    // Temporary invalidated ids management.
    pub temp_invalidated_ids_storage: RwLock<InvalidatedIdsStorage>,
    temp_invalidated_ids: RwLock<HashMap<u128, Vec<u128>>>,

    // Prevent a mutable segment from being modified after it is built.
    finalized: bool,
    last_sequence_number: AtomicU64,
    num_docs: AtomicU64,
    created_at: Instant,
}

impl MutableSegment {
    pub fn new(config: CollectionConfig, base_directory: String) -> Result<Self> {
        let temp_invalidated_ids_directory =
            format!("{}/temp_invalidated_ids_storage", base_directory);
        let mut temp_invalidated_ids_storage =
            InvalidatedIdsStorage::read(&temp_invalidated_ids_directory)?;

        // Read back invalidated ids on reset
        let mut temp_invalidated_ids = HashMap::new();
        for invalidated_id in temp_invalidated_ids_storage.iter() {
            temp_invalidated_ids
                .entry(invalidated_id.user_id)
                .or_insert_with(Vec::new)
                .push(invalidated_id.doc_id);
        }

        Ok(Self {
            multi_spann_builder: MultiSpannBuilder::new(config, base_directory)?,
            temp_invalidated_ids_storage: RwLock::new(temp_invalidated_ids_storage),
            temp_invalidated_ids: RwLock::new(temp_invalidated_ids),
            finalized: false,
            last_sequence_number: AtomicU64::new(0),
            num_docs: AtomicU64::new(0),
            created_at: Instant::now(),
        })
    }

    pub fn created_at(&self) -> Instant {
        self.created_at
    }

    pub fn insert(&mut self, doc_id: u128, data: &[f32]) -> Result<()> {
        if self.finalized {
            return Err(anyhow::anyhow!("Cannot insert into a finalized segment"));
        }

        self.insert_for_user(0, doc_id, data, 0)
    }

    /// Insert a document for a user
    pub fn insert_for_user(
        &self,
        user_id: u128,
        doc_id: u128,
        data: &[f32],
        sequence_number: u64,
    ) -> Result<()> {
        debug!(
            "Inserting for user: {}, doc_id: {}, sequence_number: {}",
            user_id, doc_id, sequence_number
        );
        if self.finalized {
            return Err(anyhow::anyhow!("Cannot insert into a finalized segment"));
        }

        self.multi_spann_builder.insert(user_id, doc_id, data)?;
        self.last_sequence_number
            .store(sequence_number, std::sync::atomic::Ordering::Relaxed);
        self.num_docs
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    pub fn remove(&self, user_id: u128, doc_id: u128) -> Result<bool> {
        // Acquire the write lock
        let mut temp_invalidated_ids = self.temp_invalidated_ids.write();
        let entry = temp_invalidated_ids.entry(user_id).or_insert_with(Vec::new);
        if entry.contains(&doc_id) {
            Ok(false)
        } else {
            // At this point we don't want to check if the doc_id exists for this user_id,
            // just register the invalidation and we'll filter out the docs upon flushing.
            entry.push(doc_id);
            self.temp_invalidated_ids_storage
                .write()
                .invalidate(user_id, doc_id)?;
            Ok(true)
        }
    }

    pub fn build(&mut self, base_directory: String, name: String) -> Result<()> {
        if self.finalized {
            return Err(anyhow::anyhow!("Cannot build a finalized segment"));
        }

        let segment_directory = format!("{}/{}", base_directory, name);
        std::fs::create_dir_all(&segment_directory)?;

        self.multi_spann_builder.build()?;

        let multi_spann_writer = MultiSpannWriter::new(segment_directory);
        multi_spann_writer.write(&mut self.multi_spann_builder)?;
        self.finalized = true;
        Ok(())
    }

    pub fn last_sequence_number(&self) -> u64 {
        self.last_sequence_number
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn num_docs(&self) -> u64 {
        self.num_docs.load(std::sync::atomic::Ordering::Relaxed)
    }
}

unsafe impl Send for MutableSegment {}

unsafe impl Sync for MutableSegment {}
