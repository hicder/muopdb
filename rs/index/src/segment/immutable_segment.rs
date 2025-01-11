use anyhow::{anyhow, Ok, Result};
use quantization::noq::noq::NoQuantizer;
use utils::distance::l2::L2DistanceCalculator;

use super::Segment;
use crate::collection::SegmentSearchable;
use crate::index::Searchable;
use crate::multi_spann::index::MultiSpannIndex;

/// This is an immutable segment. This usually contains a single index.
pub struct ImmutableSegment {
    index: MultiSpannIndex<NoQuantizer<L2DistanceCalculator>>,
}

impl ImmutableSegment {
    pub fn new(index: MultiSpannIndex<NoQuantizer<L2DistanceCalculator>>) -> Self {
        Self { index }
    }
}

impl Segment for ImmutableSegment {
    fn insert(&mut self, _doc_id: u64, _data: &[f32]) -> Result<()> {
        Err(anyhow!("ImmutableSegment does not support insertion"))
    }

    fn remove(&mut self, _doc_id: u64) -> Result<bool> {
        // TODO(hicder): Implement this
        Ok(false)
    }

    fn may_contains(&self, _doc_id: u64) -> bool {
        // TODO(hicder): Implement this
        return true;
    }
}

impl Searchable for ImmutableSegment {
    fn search(
        &self,
        query: &[f32],
        k: usize,
        ef_construction: u32,
        context: &mut crate::utils::SearchContext,
    ) -> Option<Vec<crate::utils::IdWithScore>> {
        self.index.search(query, k, ef_construction, context)
    }

    fn search_with_id(
        &self,
        id: u64,
        query: &[f32],
        k: usize,
        ef_construction: u32,
        context: &mut crate::utils::SearchContext,
    ) -> Option<Vec<crate::utils::IdWithScore>> {
        self.index
            .search_with_id(id, query, k, ef_construction, context)
    }
}

impl SegmentSearchable for ImmutableSegment {}
unsafe impl Send for ImmutableSegment {}
unsafe impl Sync for ImmutableSegment {}
