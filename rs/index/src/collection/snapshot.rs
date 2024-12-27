use std::sync::Arc;

use super::{BoxedSegmentSearchable, Collection};
use crate::index::Searchable;
use crate::utils::SearchContext;

/// Snapshot provides a view of the collection at a given point in time
pub struct Snapshot {
    pub segments: Vec<Arc<BoxedSegmentSearchable>>,
    pub version: u64,
    pub collection: Arc<Collection>,
}

impl Snapshot {
    pub fn new(
        segments: Vec<Arc<BoxedSegmentSearchable>>,
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
}

/// Search the collection using the given query
impl Searchable for Snapshot {
    fn search(
        &self,
        query: &[f32],
        k: usize,
        ef_construction: u32,
        context: &mut SearchContext,
    ) -> Option<Vec<crate::utils::IdWithScore>> {
        // Query each index, then take the top k results
        // TODO(hicder): Handle case where docs are deleted in later segments
        let mut scored_results: Vec<_> = self
            .segments
            .iter()
            .filter_map(|index| index.search(query, k, ef_construction, context))
            .flat_map(|results| results.into_iter().map(|id_score| id_score))
            .collect();

        // Sort and take the top k results
        scored_results.sort_by(|x, y| x.cmp(y));
        scored_results.truncate(k);

        Some(scored_results)
    }
}

impl Drop for Snapshot {
    fn drop(&mut self) {
        self.collection.release_version(self.version);
    }
}
