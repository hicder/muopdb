use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::time::Instant;

use anyhow::{Ok, Result};
use config::collection::CollectionConfig;
use log::debug;
use parking_lot::RwLock;

use crate::multi_spann::builder::MultiSpannBuilder;
use crate::multi_spann::writer::MultiSpannWriter;

pub struct MutableSegment {
    multi_spann_builder: MultiSpannBuilder,

    // Using AtomicBool to ensure cache coherency.
    flushing_started: AtomicBool,
    invalidated_ids_per_user: RwLock<HashMap<u128, HashSet<u128>>>,

    // Prevent a mutable segment from being modified after it is built.
    finalized: bool,
    last_sequence_number: AtomicU64,
    num_docs: AtomicU64,
    created_at: Instant,
}

impl MutableSegment {
    pub fn new(config: CollectionConfig, base_directory: String) -> Result<Self> {
        Ok(Self {
            multi_spann_builder: MultiSpannBuilder::new(config, base_directory)?,
            flushing_started: AtomicBool::new(false),
            invalidated_ids_per_user: RwLock::new(HashMap::new()),
            finalized: false,
            last_sequence_number: AtomicU64::new(0),
            num_docs: AtomicU64::new(0),
            created_at: Instant::now(),
        })
    }

    pub fn created_at(&self) -> Instant {
        self.created_at
    }

    pub fn insert(&self, doc_id: u128, data: &[f32]) -> Result<()> {
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

        if self
            .flushing_started
            .load(std::sync::atomic::Ordering::Acquire)
        {
            if let Some(invalidated_id_set) =
                self.invalidated_ids_per_user.write().get_mut(&user_id)
            {
                // It's ok if the doc_id hasn't been invalidated before
                invalidated_id_set.remove(&doc_id);
            }
        }
        self.multi_spann_builder.insert(user_id, doc_id, data)?;
        self.last_sequence_number
            .store(sequence_number, std::sync::atomic::Ordering::Relaxed);
        self.num_docs
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    pub fn invalidate(&self, user_id: u128, doc_id: u128) -> Result<bool> {
        if self
            .flushing_started
            .load(std::sync::atomic::Ordering::Acquire)
        {
            let mut invalidated_map = self.invalidated_ids_per_user.write();
            // Get the entry for the doc_id, or insert a new empty HashSet if it doesn't exist
            let invalidated_set = invalidated_map.entry(user_id).or_insert_with(HashSet::new);

            // Insert the element into the HashSet
            Ok(invalidated_set.insert(doc_id))
        } else {
            self.multi_spann_builder.invalidate(user_id, doc_id)
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

    pub fn start_being_flushed(&mut self) {
        self.flushing_started
            .store(true, std::sync::atomic::Ordering::Release);
    }

    pub fn invalidated_ids_map(&self) -> &RwLock<HashMap<u128, HashSet<u128>>> {
        &self.invalidated_ids_per_user
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

#[cfg(test)]
mod tests {
    use config::collection::CollectionConfig;

    use super::*;

    #[tokio::test]
    async fn test_mutable_segment() {
        let tmp_dir = tempdir::TempDir::new("mutable_segment_test").unwrap();
        let base_dir = tmp_dir.path().to_str().unwrap().to_string();

        let segment_config = CollectionConfig::default_test_config();
        let mut mutable_segment = MutableSegment::new(segment_config.clone(), base_dir)
            .expect("Failed to create mutable segment");

        assert!(mutable_segment.insert(0, &[1.0, 2.0, 3.0, 4.0]).is_ok());
        assert!(mutable_segment.insert(1, &[5.0, 6.0, 7.0, 8.0]).is_ok());
        assert!(mutable_segment.insert(2, &[9.0, 10.0, 11.0, 12.0]).is_ok());

        mutable_segment.start_being_flushed();

        assert!(mutable_segment
            .invalidate(0, 0)
            .expect("Failed to invalidate"));
        assert!(!mutable_segment
            .invalidate(0, 0)
            .expect("Failed to invalidate"));
        assert!(mutable_segment.insert(0, &[5.0, 6.0, 7.0, 8.0]).is_ok());
        assert!(mutable_segment
            .invalidate(0, 0)
            .expect("Failed to invalidate"));

        let invalidated_map = mutable_segment.invalidated_ids_per_user.read();
        let invalidated_set = invalidated_map.get(&0);
        assert!(invalidated_set.is_some());
        assert_eq!(invalidated_set.unwrap().len(), 1);
        assert!(invalidated_set.unwrap().contains(&0));
    }
}
