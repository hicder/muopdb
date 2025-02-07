use anyhow::{anyhow, Ok, Result};
use quantization::quantization::Quantizer;

use super::Segment;
use crate::collection::SegmentSearchable;
use crate::index::Searchable;
use crate::multi_spann::index::MultiSpannIndex;
use crate::spann::iter::SpannIter;

/// This is an immutable segment. This usually contains a single index.
pub struct ImmutableSegment<Q: Quantizer> {
    index: MultiSpannIndex<Q>,
    name: String,
}

impl<Q: Quantizer> ImmutableSegment<Q> {
    pub fn new(index: MultiSpannIndex<Q>, name: String) -> Self {
        Self { index, name }
    }

    pub fn user_ids(&self) -> Vec<u128> {
        self.index.user_ids()
    }

    pub fn iter_for_user(&self, user_id: u128) -> Option<SpannIter<Q>> {
        self.index.iter_for_user(user_id)
    }
}

/// This is the implementation of Segment for ImmutableSegment.
impl<Q: Quantizer> Segment for ImmutableSegment<Q> {
    /// ImmutableSegment does not support insertion.
    fn insert(&self, _doc_id: u64, _data: &[f32]) -> Result<()> {
        Err(anyhow!("ImmutableSegment does not support insertion"))
    }

    /// ImmutableSegment does not support removal.
    fn remove(&self, _doc_id: u64) -> Result<bool> {
        // TODO(hicder): Implement this
        Ok(false)
    }

    /// ImmutableSegment does not support contains.
    fn may_contains(&self, _doc_id: u64) -> bool {
        // TODO(hicder): Implement this
        return true;
    }

    fn name(&self) -> String {
        self.name.clone()
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
