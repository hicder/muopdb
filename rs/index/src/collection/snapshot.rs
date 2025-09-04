use std::sync::Arc;

use quantization::noq::noq::NoQuantizerL2;
use quantization::pq::pq::ProductQuantizerL2;
use quantization::quantization::Quantizer;

use super::core::Collection;
use crate::segment::BoxedImmutableSegment;
use crate::utils::SearchResult;

/// Snapshot provides a view of the collection at a given point in time
pub struct Snapshot<Q: Quantizer + Clone + Send + Sync + 'static> {
    pub segments: Vec<BoxedImmutableSegment<Q>>,
    pub version: u64,
    pub collection: Arc<Collection<Q>>,
}

impl<Q: Quantizer + Clone + Send + Sync + 'static> Snapshot<Q> {
    pub fn new(
        segments: Vec<BoxedImmutableSegment<Q>>,
        version: u64,
        collection: Arc<Collection<Q>>,
    ) -> Self {
        Self {
            segments,
            version,
            collection,
        }
    }

    pub fn version(&self) -> u64 {
        self.version
    }

    pub async fn search_for_users(
        snapshot: Arc<Snapshot<Q>>,
        user_ids: &[u128],
        query: Vec<f32>,
        k: usize,
        ef_construction: u32,
        record_pages: bool,
    ) -> Option<SearchResult>
    where
        <Q as Quantizer>::QuantizedT: Send + Sync,
    {
        let mut results = SearchResult::new();
        for user_id in user_ids {
            match snapshot
                .search_for_user(*user_id, query.clone(), k, ef_construction, record_pages)
                .await
            {
                Some(id_results) => {
                    results.id_with_scores.extend(id_results.id_with_scores);
                    results.stats.merge(&id_results.stats);
                }
                None => {}
            }
        }

        results.id_with_scores.sort();
        results.id_with_scores.truncate(k);

        Some(results)
    }
}

/// Search the collection using the given query
impl<Q: Quantizer + Clone + Send + Sync + 'static> Snapshot<Q> {
    pub async fn search_for_user(
        &self,
        user_id: u128,
        query: Vec<f32>,
        k: usize,
        ef_construction: u32,
        record_pages: bool,
    ) -> Option<SearchResult>
    where
        <Q as Quantizer>::QuantizedT: Send + Sync,
    {
        // Query each index, then take the top k results
        // TODO(hicder): Handle case where docs are deleted in later segments

        let mut scored_results = SearchResult::new();
        for segment in &self.segments {
            let s = segment.clone();
            let q = query.clone();
            if let Some(results) = BoxedImmutableSegment::search_with_id(
                s,
                user_id,
                q,
                k,
                ef_construction,
                record_pages,
            )
            .await
            {
                scored_results.id_with_scores.extend(results.id_with_scores);
                scored_results.stats.merge(&results.stats);
            }
        }

        // Sort and take the top k results
        scored_results.id_with_scores.sort();
        scored_results.id_with_scores.truncate(k);

        Some(scored_results)
    }
}

impl<Q: Quantizer + Clone + Send + Sync + 'static> Drop for Snapshot<Q> {
    fn drop(&mut self) {
        self.collection.release_version(self.version);
    }
}

pub enum SnapshotWithQuantizer {
    SnapshotNoQuantizer(Arc<Snapshot<NoQuantizerL2>>),
    SnapshotProductQuantizer(Arc<Snapshot<ProductQuantizerL2>>),
}

impl SnapshotWithQuantizer {
    pub fn new_with_no_quantizer(snapshot: Snapshot<NoQuantizerL2>) -> Self {
        Self::SnapshotNoQuantizer(Arc::new(snapshot))
    }

    pub fn new_with_product_quantizer(snapshot: Snapshot<ProductQuantizerL2>) -> Self {
        Self::SnapshotProductQuantizer(Arc::new(snapshot))
    }

    pub async fn search_for_users(
        snapshot: SnapshotWithQuantizer,
        user_ids: &[u128],
        query: Vec<f32>,
        k: usize,
        ef_construction: u32,
        record_pages: bool,
    ) -> Option<SearchResult> {
        match snapshot {
            Self::SnapshotNoQuantizer(snapshot) => {
                Snapshot::<NoQuantizerL2>::search_for_users(
                    snapshot,
                    user_ids,
                    query.clone(),
                    k,
                    ef_construction,
                    record_pages,
                )
                .await
            }
            Self::SnapshotProductQuantizer(snapshot) => {
                Snapshot::<ProductQuantizerL2>::search_for_users(
                    snapshot,
                    user_ids,
                    query.clone(),
                    k,
                    ef_construction,
                    record_pages,
                )
                .await
            }
        }
    }
}
