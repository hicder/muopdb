use anyhow::{anyhow, Ok, Result};
use quantization::quantization::Quantizer;

use super::Segment;
use crate::collection::SegmentSearchable;
use crate::index::Searchable;
use crate::multi_spann::index::MultiSpannIndex;

/// This is an immutable segment. This usually contains a single index.
pub struct ImmutableSegment<Q: Quantizer> {
    index: MultiSpannIndex<Q>,
}

impl<Q: Quantizer> ImmutableSegment<Q> {
    pub fn new(index: MultiSpannIndex<Q>) -> Self {
        Self { index }
    }
}

impl<Q: Quantizer> Segment for ImmutableSegment<Q> {
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

impl<Q: Quantizer> Searchable for ImmutableSegment<Q> {
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
        id: u128,
        query: &[f32],
        k: usize,
        ef_construction: u32,
        context: &mut crate::utils::SearchContext,
    ) -> Option<Vec<crate::utils::IdWithScore>> {
        self.index
            .search_with_id(id, query, k, ef_construction, context)
    }
}

impl<Q: Quantizer> SegmentSearchable for ImmutableSegment<Q> {}
unsafe impl<Q: Quantizer> Send for ImmutableSegment<Q> {}
unsafe impl<Q: Quantizer> Sync for ImmutableSegment<Q> {}
