use anyhow::{anyhow, Ok, Result};

use super::Segment;
use crate::collection::SegmentSearchable;
use crate::index::Searchable;
use crate::spann::index::Spann;

/// This is an immutable segment. This usually contains a single index.
pub struct ImmutableSegment {
    index: Spann,
}

impl ImmutableSegment {
    pub fn new(index: Spann) -> Self {
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
}

impl SegmentSearchable for ImmutableSegment {}
