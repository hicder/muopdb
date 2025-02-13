use std::sync::Arc;

use quantization::noq::noq::NoQuantizerL2;
use quantization::pq::pq::ProductQuantizerL2;
use quantization::quantization::Quantizer;

use super::collection::Collection;
use crate::segment::BoxedImmutableSegment;
use crate::utils::{IdWithScore, SearchContext};

/// Snapshot provides a view of the collection at a given point in time
pub struct Snapshot<Q: Quantizer + Clone + Send + Sync> {
    pub segments: Vec<BoxedImmutableSegment<Q>>,
    pub version: u64,
    pub collection: Arc<Collection<Q>>,
}

impl<Q: Quantizer + Clone + Send + Sync> Snapshot<Q> {
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

    pub async fn search_for_ids(
        &self,
        ids: &[u128],
        query: &[f32],
        k: usize,
        ef_construction: u32,
        context: &mut SearchContext,
    ) -> Option<Vec<IdWithScore>>
    where
        <Q as Quantizer>::QuantizedT: Send + Sync,
    {
        let mut results: Vec<IdWithScore> = vec![];
        for id in ids {
            match self
                .search_with_id(*id, query, k, ef_construction, context)
                .await
            {
                Some(id_results) => {
                    results.extend(id_results);
                }
                None => {}
            }
        }

        results.sort();
        results.truncate(k);

        Some(results)
    }
}

/// Search the collection using the given query
impl<Q: Quantizer + Clone + Send + Sync> Snapshot<Q> {
    pub async fn search_with_id(
        &self,
        id: u128,
        query: &[f32],
        k: usize,
        ef_construction: u32,
        context: &mut SearchContext,
    ) -> Option<Vec<IdWithScore>>
    where
        <Q as Quantizer>::QuantizedT: Send + Sync,
    {
        // Query each index, then take the top k results
        // TODO(hicder): Handle case where docs are deleted in later segments

        let mut scored_results = Vec::new();
        for segment in &self.segments {
            if let Some(results) = segment
                .search_with_id(id, query, k, ef_construction, context)
                .await
            {
                scored_results.extend(results);
            }
        }

        // Sort and take the top k results
        scored_results.sort();
        scored_results.truncate(k);

        Some(scored_results)
    }
}

impl<Q: Quantizer + Clone + Send + Sync> Drop for Snapshot<Q> {
    fn drop(&mut self) {
        self.collection.release_version(self.version);
    }
}

pub enum SnapshotWithQuantizer {
    SnapshotNoQuantizer(Snapshot<NoQuantizerL2>),
    SnapshotProductQuantizer(Snapshot<ProductQuantizerL2>),
}

impl SnapshotWithQuantizer {
    pub fn new_with_no_quantizer(snapshot: Snapshot<NoQuantizerL2>) -> Self {
        Self::SnapshotNoQuantizer(snapshot)
    }

    pub fn new_with_product_quantizer(snapshot: Snapshot<ProductQuantizerL2>) -> Self {
        Self::SnapshotProductQuantizer(snapshot)
    }

    pub async fn search_for_ids(
        &self,
        ids: &[u128],
        query: &[f32],
        k: usize,
        ef_construction: u32,
        context: &mut SearchContext,
    ) -> Option<Vec<IdWithScore>> {
        match self {
            Self::SnapshotNoQuantizer(snapshot) => {
                snapshot
                    .search_for_ids(ids, query, k, ef_construction, context)
                    .await
            }
            Self::SnapshotProductQuantizer(snapshot) => {
                snapshot
                    .search_for_ids(ids, query, k, ef_construction, context)
                    .await
            }
        }
    }
}
