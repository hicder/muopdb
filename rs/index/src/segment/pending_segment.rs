use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;

use anyhow::{anyhow, Result};
use config::collection::CollectionConfig;
use parking_lot::RwLock;
use quantization::quantization::Quantizer;

use super::{BoxedImmutableSegment, Segment};
use crate::ivf::files::invalidated_ids::InvalidatedIdsStorage;
use crate::multi_spann::index::MultiSpannIndex;
use crate::multi_spann::reader::MultiSpannReader;
use crate::utils::SearchResult;

/// Represents an intermediate segment used during optimization processes (like merging or vacuuming).
///
/// A `PendingSegment` wraps one or more existing immutable segments and provides a mutable
/// view for optimization operations. Initially, it delegates search queries to its inner segments
/// and records invalidation operations (removals) in temporary storage.
///
/// Once the optimization process is complete, the `PendingSegment` builds an internal index
/// from the optimized data. At this point, the `use_internal_index` flag is set to true,
/// and subsequent search queries are handled by the internal index, with the recorded
/// invalidations applied to it.
///
/// This design allows for concurrent reads (searches) on the original segments while the
/// optimization is in progress, and then seamlessly switches to the optimized internal index
/// once it's ready.
///
/// **Locking:**
/// - `temp_invalidated_ids_storage`: Protected by a `RwLock` for concurrent read/write access to the on-disk storage.
/// - `temp_invalidated_ids`: Protected by a `RwLock` for concurrent read/write access to the in-memory map of invalidated IDs.
/// - `index`: Protected by a `RwLock` for concurrent read/write access to the internal index once built.
/// - `use_internal_index`: An `AtomicBool` is used to signal when the internal index is ready and should be used for searches, ensuring visibility across threads.
pub struct PendingSegment<Q: Quantizer + Clone + Send + Sync> {
    inner_segments: Vec<BoxedImmutableSegment<Q>>,
    inner_segments_names: Vec<String>,
    name: String,
    parent_directory: String,

    // Whether to use the internal index instead of passing the query to inner segments.
    // Invariant: use_internal_index is true if and only if index is Some.
    // Use AtomicBool to ensure cache coherency, making sure latest updates are always propagated
    // correctly to subsequent reads.
    use_internal_index: AtomicBool,

    // Temporary invalidated ids management, only used when use_internal_index is false.
    temp_invalidated_ids_storage: RwLock<InvalidatedIdsStorage>,
    temp_invalidated_ids: RwLock<HashMap<u128, Vec<u128>>>,

    // The internal index.
    index: RwLock<Option<MultiSpannIndex<Q>>>,

    collection_config: CollectionConfig,
}

impl<Q: Quantizer + Clone + Send + Sync> PendingSegment<Q> {
    pub fn new(
        inner_segments: Vec<BoxedImmutableSegment<Q>>,
        data_directory: String,
        collection_config: CollectionConfig,
    ) -> Self {
        let path = PathBuf::from(&data_directory);
        // name is the last portion of the data_directory
        let name = path.file_name().unwrap().to_str().unwrap().to_string();

        // base directory is the directory of the data_directory
        let parent_directory = path.parent().unwrap().to_str().unwrap().to_string();
        let inner_segments_names = inner_segments
            .iter()
            .map(|segment| segment.name())
            .collect();

        let temp_invalidated_ids_directory =
            format!("{data_directory}/temp_invalidated_ids_storage");
        // TODO(tyb) avoid unwrap here
        let temp_invalidated_ids_storage =
            InvalidatedIdsStorage::read(&temp_invalidated_ids_directory).unwrap();

        let mut temp_invalidated_ids = HashMap::new();
        for invalidated_id in temp_invalidated_ids_storage.iter() {
            temp_invalidated_ids
                .entry(invalidated_id.user_id)
                .or_insert_with(Vec::new)
                .push(invalidated_id.doc_id);
        }

        Self {
            inner_segments,
            inner_segments_names,
            name,
            parent_directory,
            index: RwLock::new(None),
            use_internal_index: AtomicBool::new(false),
            // TODO(tyb): avoid unwrap here
            temp_invalidated_ids_storage: RwLock::new(temp_invalidated_ids_storage),
            temp_invalidated_ids: RwLock::new(temp_invalidated_ids),
            collection_config,
        }
    }

    /// Builds the internal `MultiSpannIndex` for the pending segment.
    ///
    /// This function reads the data files generated during the optimization process
    /// from the pending segment's directory and constructs an in-memory index.
    /// This index will be used for subsequent search operations once the
    /// `switch_to_internal_index` method is called.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The internal index has already been built (`use_internal_index` is true).
    /// - Reading the index data from disk fails.
    pub fn build_index(&self) -> Result<()> {
        if self
            .use_internal_index
            .load(std::sync::atomic::Ordering::Acquire)
        {
            // We shouldn't build the index if it already exists.
            return Err(anyhow!("Index already exists"));
        }

        let current_directory = format!("{}/{}", self.parent_directory, self.name);
        let reader = MultiSpannReader::new(current_directory);
        let index = reader.read::<Q>(
            self.collection_config.posting_list_encoding_type.clone(),
            self.collection_config.num_features,
        )?;
        self.index.write().replace(index);
        Ok(())
    }

    // Caller must hold the write lock before calling this function.
    pub fn apply_pending_deletions(&self) -> Result<()> {
        let internal_index = self.index.read();
        match &*internal_index {
            Some(index) => {
                let invalidated_ids_directory = PathBuf::from(format!(
                    "{}/invalidated_ids_storage",
                    index.base_directory()
                ));
                if !invalidated_ids_directory.exists() {
                    return Err(anyhow!("Invalidated ids directory does not exist"));
                }
                if !invalidated_ids_directory.is_dir() {
                    return Err(anyhow!("Invalidated ids path is not a directory"));
                }

                // At this point there should be no invalidated ids recorded in internal index.
                let is_empty = std::fs::read_dir(&invalidated_ids_directory)?
                    .next()
                    .is_none();
                if !is_empty {
                    return Err(anyhow!(
                        "Invalidated ids directory for internal index is not empty"
                    ));
                }

                // TODO(tyb): hard link the storage? But still need to invalidate in the hash set
                //
                // doc_id may have been removed during optimizer run (e.g. we don't add
                // invalidated docs when merging), so we just need to make sure
                // invalidating doesn't result in error, no need to verify the effectively
                // invalidated element count.
                let _ = index.invalidate_batch(&self.temp_invalidated_ids.read())?;
                Ok(())
            }
            None => Err(anyhow!("Internal index does not exist")),
        }
    }

    // Caller must hold the write lock before calling this function.
    pub fn switch_to_internal_index(&mut self) {
        self.use_internal_index
            .store(true, std::sync::atomic::Ordering::Release);
    }

    pub fn inner_segments_names(&self) -> &Vec<String> {
        &self.inner_segments_names
    }

    pub fn parent_directory(&self) -> &String {
        &self.parent_directory
    }

    pub fn base_directory(&self) -> String {
        format!("{}/{}", self.parent_directory, self.name)
    }

    pub fn inner_segments(&self) -> &Vec<BoxedImmutableSegment<Q>> {
        &self.inner_segments
    }

    pub fn collection_config(&self) -> &CollectionConfig {
        &self.collection_config
    }

    pub fn all_user_ids(&self) -> Vec<u128> {
        let mut user_ids = HashSet::new();
        for segment in &self.inner_segments {
            user_ids.extend(segment.user_ids());
        }
        user_ids.into_iter().collect()
    }

    pub fn temp_invalidated_ids_storage_directory(&self) -> String {
        self.temp_invalidated_ids_storage
            .read()
            .base_directory()
            .to_string()
    }
}

#[allow(unused)]
impl<Q: Quantizer + Clone + Send + Sync> Segment for PendingSegment<Q> {
    fn insert(&self, doc_id: u128, data: &[f32]) -> Result<()> {
        Err(anyhow::anyhow!("Pending segment does not support insert"))
    }

    fn remove(&self, user_id: u128, doc_id: u128) -> Result<bool> {
        if !self
            .use_internal_index
            .load(std::sync::atomic::Ordering::Acquire)
        {
            // No need to check inner segments to avoid complexity when a doc_id is removed from
            // one of the segment but not the other, e.g.
            // - invalidate doc_id from segment A,
            // - insert doc_id back, flush, creating segment B,
            // (doc_id is valid in B but invalidated in A)
            //
            // Adding invalidation to the temporary map + storage guarantees to always be correct.

            // Inner segments are being used to build the internal index, so they should not be
            // changed. Use temporary storage and hash map instead.

            // Acquire the write lock
            let mut temp_invalidated_ids = self.temp_invalidated_ids.write();
            let entry = temp_invalidated_ids.entry(user_id).or_default();
            if entry.contains(&doc_id) {
                Ok(false)
            } else {
                entry.push(doc_id);
                // Only write to storage if doc_id is found
                self.temp_invalidated_ids_storage
                    .write()
                    .invalidate(user_id, doc_id)?;
                Ok(true)
            }
        } else {
            let index = self.index.read();
            match &*index {
                Some(index) => index.invalidate(user_id, doc_id),
                None => unreachable!("Index should not be None if use_internal_index is set"),
            }
        }
    }

    fn may_contain(&self, _doc_id: u128) -> bool {
        false
    }

    fn name(&self) -> String {
        self.name.clone()
    }
}

#[allow(clippy::await_holding_lock)]
impl<Q: Quantizer + Clone + Send + Sync + 'static> PendingSegment<Q> {
    pub async fn search_with_id(
        &self,
        id: u128,
        query: Vec<f32>,
        k: usize,
        ef_construction: u32,
        record_pages: bool,
    ) -> Option<SearchResult>
    where
        <Q as Quantizer>::QuantizedT: Send + Sync,
    {
        if !self
            .use_internal_index
            .load(std::sync::atomic::Ordering::Acquire)
        {
            let mut results = SearchResult::new();
            // The invalidated ids vector should be very small, we can just clone it
            let invalidated_ids = self
                .temp_invalidated_ids
                .read()
                .get(&id)
                .filter(|vec| !vec.is_empty())
                .cloned()
                .unwrap_or_default();
            for segment in &self.inner_segments {
                let s = segment.clone();
                let segment_result = BoxedImmutableSegment::search_with_id(
                    s,
                    id,
                    query.clone(),
                    k + invalidated_ids.len(),
                    ef_construction,
                    record_pages,
                    None,
                )
                .await;
                if let Some(mut result) = segment_result {
                    // Filter out invalidated IDs
                    result
                        .id_with_scores
                        .retain(|id_with_score| !invalidated_ids.contains(&id_with_score.doc_id));

                    results.id_with_scores.extend(result.id_with_scores);
                    results.stats.merge(&result.stats);
                }
            }
            Some(results)
        } else {
            let index = self.index.read();
            match &*index {
                Some(index) => {
                    index
                        .search_for_user(id, query.clone(), k, ef_construction, record_pages, None)
                        .await
                }
                None => unreachable!("Index should not be None if use_internal_index is set"),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use config::collection::CollectionConfig;
    use config::enums::IntSeqEncodingType;
    use quantization::noq::noq::{NoQuantizer, NoQuantizerL2};
    use rand::Rng;
    use utils::distance::l2::L2DistanceCalculator;

    use super::*;
    use crate::multi_spann::builder::MultiSpannBuilder;
    use crate::multi_spann::writer::MultiSpannWriter;
    use crate::segment::immutable_segment::ImmutableSegment;

    fn build_segment(base_directory: String, starting_doc_id: u128) -> Result<()> {
        let mut starting_doc_id = starting_doc_id;
        let mut spann_builder_config = CollectionConfig::default_test_config();
        spann_builder_config.num_features = 4;
        spann_builder_config.initial_num_centroids = 1;
        let mut multi_spann_builder =
            MultiSpannBuilder::new(spann_builder_config, base_directory.clone())?;

        multi_spann_builder.insert(starting_doc_id % 2, starting_doc_id, &[1.0, 2.0, 3.0, 4.0])?;
        starting_doc_id += 1;
        multi_spann_builder.insert(starting_doc_id % 2, starting_doc_id, &[5.0, 6.0, 7.0, 8.0])?;
        starting_doc_id += 1;
        multi_spann_builder.insert(
            starting_doc_id % 2,
            starting_doc_id,
            &[9.0, 10.0, 11.0, 12.0],
        )?;
        multi_spann_builder.build()?;

        let multi_spann_writer = MultiSpannWriter::new(base_directory.clone());
        multi_spann_writer.write(&mut multi_spann_builder)?;
        Ok(())
    }

    fn read_segment(base_directory: String) -> Result<MultiSpannIndex<NoQuantizerL2>> {
        let reader = MultiSpannReader::new(base_directory);
        let index = reader.read::<NoQuantizerL2>(IntSeqEncodingType::PlainEncoding, 4)?;
        Ok(index)
    }

    #[tokio::test]
    async fn test_pending_segment() -> Result<()> {
        // temp directory
        let tmp_dir = tempdir::TempDir::new("pending_segment_test").unwrap();
        let base_dir = tmp_dir.path().to_str().unwrap().to_string();

        // Create dir for segment1
        let segment1_dir = format!("{base_dir}/segment_1");
        std::fs::create_dir_all(segment1_dir.clone()).unwrap();
        build_segment(segment1_dir.clone(), 0)?;
        let segment1 = read_segment(segment1_dir.clone())?;
        let segment1 = BoxedImmutableSegment::<NoQuantizer<L2DistanceCalculator>>::FinalizedSegment(
            Arc::new(RwLock::new(ImmutableSegment::new(
                segment1,
                "segment_1".to_string(),
                None,
            ))),
        );

        let random_name = format!(
            "pending_segment_{}",
            rand::thread_rng().gen_range(0..1000000)
        );
        let pending_dir = format!("{base_dir}/{random_name}");
        std::fs::create_dir_all(pending_dir.clone()).unwrap();

        // Create a pending segment
        let pending_segment = PendingSegment::<NoQuantizer<L2DistanceCalculator>>::new(
            vec![segment1],
            pending_dir.clone(),
            CollectionConfig::default_test_config(),
        );

        let results = pending_segment
            .search_with_id(0, vec![1.0, 2.0, 3.0, 4.0], 1, 10, false)
            .await;
        let res = results.unwrap();
        assert_eq!(res.id_with_scores.len(), 1);
        assert_eq!(res.id_with_scores[0].doc_id, 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_pending_segment_remove() -> Result<()> {
        // temp directory
        let tmp_dir = tempdir::TempDir::new("pending_segment_remove_test").unwrap();
        let base_dir = tmp_dir.path().to_str().unwrap().to_string();

        // Create dir for segment1
        let segment1_dir = format!("{base_dir}/segment_1");
        std::fs::create_dir_all(segment1_dir.clone()).unwrap();
        build_segment(segment1_dir.clone(), 0)?;
        let segment1 = read_segment(segment1_dir.clone())?;
        let segment1 = BoxedImmutableSegment::<NoQuantizer<L2DistanceCalculator>>::FinalizedSegment(
            Arc::new(RwLock::new(ImmutableSegment::new(
                segment1,
                "segment_1".to_string(),
                None,
            ))),
        );

        let random_name = format!(
            "pending_segment_{}",
            rand::thread_rng().gen_range(0..1000000)
        );
        let pending_dir = format!("{base_dir}/{random_name}");
        std::fs::create_dir_all(pending_dir.clone()).unwrap();

        // Create a pending segment
        let pending_segment = PendingSegment::<NoQuantizer<L2DistanceCalculator>>::new(
            vec![segment1],
            pending_dir.clone(),
            CollectionConfig::default_test_config(),
        );

        assert!(pending_segment.remove(0, 0)?);

        let results = pending_segment
            .search_with_id(0, vec![1.0, 2.0, 3.0, 4.0], 1, 10, false)
            .await;
        let res = results.unwrap();
        assert_eq!(res.id_with_scores.len(), 1);
        assert_eq!(res.id_with_scores[0].doc_id, 2);

        Ok(())
    }

    #[tokio::test]
    async fn test_pending_segment_invalidated() -> Result<()> {
        // temp directory
        let tmp_dir = tempdir::TempDir::new("pending_segment_invalidated_test").unwrap();
        let base_dir = tmp_dir.path().to_str().unwrap().to_string();

        // Create dir for segment1
        let segment1_dir = format!("{base_dir}/segment_1");
        std::fs::create_dir_all(segment1_dir.clone()).unwrap();
        build_segment(segment1_dir.clone(), 0)?;
        let segment1 = read_segment(segment1_dir.clone())?;
        let segment1 = BoxedImmutableSegment::<NoQuantizer<L2DistanceCalculator>>::FinalizedSegment(
            Arc::new(RwLock::new(ImmutableSegment::new(
                segment1,
                "segment_1".to_string(),
                None,
            ))),
        );

        let random_name = format!(
            "pending_segment_{}",
            rand::thread_rng().gen_range(0..1000000)
        );
        let pending_dir = format!("{base_dir}/{random_name}");

        let invalidated_ids_dir = format!("{pending_dir}/temp_invalidated_ids_storage");

        assert!(std::fs::create_dir_all(&invalidated_ids_dir).is_ok());

        let mut storage = InvalidatedIdsStorage::new(&invalidated_ids_dir, 1024);

        // Invalidate a user ID and doc ID
        assert!(storage.invalidate(0, 0).is_ok());

        // Create a pending segment
        let pending_segment = PendingSegment::<NoQuantizer<L2DistanceCalculator>>::new(
            vec![segment1],
            pending_dir.clone(),
            CollectionConfig::default_test_config(),
        );

        let results = pending_segment
            .search_with_id(0, vec![1.0, 2.0, 3.0, 4.0], 1, 10, false)
            .await;
        let res = results.unwrap();
        assert_eq!(res.id_with_scores.len(), 1);
        assert_eq!(res.id_with_scores[0].doc_id, 2);

        Ok(())
    }
}
