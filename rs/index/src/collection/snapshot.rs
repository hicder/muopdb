use std::sync::Arc;

use parking_lot::Mutex;
use quantization::noq::noq::NoQuantizerL2;
use quantization::pq::pq::ProductQuantizerL2;
use quantization::quantization::Quantizer;

use super::collection::Collection;
use crate::segment::BoxedImmutableSegment;
use crate::utils::{IdWithScore, SearchContext};

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

    pub async fn search_for_ids(
        snapshot: Arc<Snapshot<Q>>,
        ids: &[u128],
        query: Vec<f32>,
        k: usize,
        ef_construction: u32,
        context: Arc<Mutex<SearchContext>>,
    ) -> Option<Vec<IdWithScore>>
    where
        <Q as Quantizer>::QuantizedT: Send + Sync,
    {
        let mut results: Vec<IdWithScore> = vec![];
        for id in ids {
            match snapshot
                .search_with_id(*id, query.clone(), k, ef_construction, context.clone())
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
impl<Q: Quantizer + Clone + Send + Sync+ 'static> Snapshot<Q> {
    pub async fn search_with_id(
        &self,
        id: u128,
        query: Vec<f32>,
        k: usize,
        ef_construction: u32,
        context: Arc<Mutex<SearchContext>>,
    ) -> Option<Vec<IdWithScore>>
    where
        <Q as Quantizer>::QuantizedT: Send + Sync,
    {
        // Query each index, then take the top k results
        // TODO(hicder): Handle case where docs are deleted in later segments

        let mut scored_results = Vec::new();
        for segment in &self.segments {
            let s = segment.clone();
            let q = query.clone();
            let context = context.clone();
            if let Some(results) = 
            BoxedImmutableSegment::search_with_id(s, id, q, k, ef_construction, context.clone())
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

    pub async fn search_for_ids(
        snapshot: SnapshotWithQuantizer,
        ids: &[u128],
        query: Vec<f32>,
        k: usize,
        ef_construction: u32,
        context: Arc<Mutex<SearchContext>>,
    ) -> Option<Vec<IdWithScore>> {
        match snapshot {
            Self::SnapshotNoQuantizer(snapshot) => {
                Snapshot::<NoQuantizerL2>::search_for_ids(snapshot, ids, query.clone(), k, ef_construction, context)
                    .await
            }
            Self::SnapshotProductQuantizer(snapshot) => {
                Snapshot::<ProductQuantizerL2>::search_for_ids(snapshot, ids, query.clone(), k, ef_construction, context)
                    .await
            }
        }
    }
}
