use std::sync::Arc;

use super::Collection;
use crate::index::Searchable;
use crate::segment::BoxedImmutableSegment;
use crate::utils::{IdWithScore, SearchContext};

/// Snapshot provides a view of the collection at a given point in time
pub struct Snapshot {
    pub segments: Vec<BoxedImmutableSegment>,
    pub version: u64,
    pub collection: Arc<Collection>,
}

impl Snapshot {
    pub fn new(
        segments: Vec<BoxedImmutableSegment>,
        version: u64,
        collection: Arc<Collection>,
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

    pub fn search_for_ids(
        &self,
        ids: &[u128],
        query: &[f32],
        k: usize,
        ef_construction: u32,
        context: &mut SearchContext,
    ) -> Option<Vec<IdWithScore>> {
        let mut results: Vec<IdWithScore> = vec![];
        for id in ids {
            match self.search_with_id(*id, query, k, ef_construction, context) {
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
impl Searchable for Snapshot {
    fn search_with_id(
        &self,
        id: u128,
        query: &[f32],
        k: usize,
        ef_construction: u32,
        context: &mut SearchContext,
    ) -> Option<Vec<IdWithScore>> {
        // Query each index, then take the top k results
        // TODO(hicder): Handle case where docs are deleted in later segments
        let mut scored_results: Vec<_> = self
            .segments
            .iter()
            .filter_map(|index| index.search_with_id(id, query, k, ef_construction, context))
            .flat_map(|results| results.into_iter().map(|id_score| id_score))
            .collect();

        // Sort and take the top k results
        scored_results.sort_by(|x, y| x.cmp(y));
        scored_results.truncate(k);

        Some(scored_results)
    }

    fn search(
        &self,
        query: &[f32],
        k: usize,
        ef_construction: u32,
        context: &mut SearchContext,
    ) -> Option<Vec<IdWithScore>> {
        self.search_with_id(0u128, query, k, ef_construction, context)
    }
}

impl Drop for Snapshot {
    fn drop(&mut self) {
        self.collection.release_version(self.version);
    }
}
