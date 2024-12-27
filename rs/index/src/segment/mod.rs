pub mod immutable_segment;

use anyhow::Result;

pub trait Segment {
    /// Insert a document into the segment.
    /// Returns an error if the insertion process fails for any reason.
    /// NOTE: Some type of segment may not support insertion.
    fn insert(&mut self, doc_id: u64, data: &[f32]) -> Result<()>;

    /// Remove a document from the segment.
    /// Returns true if the document was removed, false if the document was not found.
    /// Returns an error if the removal process fails for any reason.
    fn remove(&mut self, doc_id: u64) -> Result<bool>;

    /// Returns true if the segment may contain the given document.
    /// False if the segment definitely does not contain the document.
    fn may_contains(&self, doc_id: u64) -> bool;
}
