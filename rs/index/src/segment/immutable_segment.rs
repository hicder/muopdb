use super::Segment;
use anyhow::{anyhow, Ok, Result};

/// This is an immutable segment. This usually contains a single index.
pub struct ImmutableSegment {}

impl ImmutableSegment {
    pub fn new() -> Self {
        Self {}
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
