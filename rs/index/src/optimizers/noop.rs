use anyhow::Result;
use fs_extra::dir::CopyOptions;
use quantization::quantization::Quantizer;

use super::SegmentOptimizer;
use crate::segment::pending_segment::PendingSegment;
use crate::segment::Segment;

pub struct NoopOptimizer<Q: Quantizer + Clone> {
    _marker: std::marker::PhantomData<Q>,
}

/// This optimizer does nothing. It just copies the original segment to a new segment.
/// Useful for testing the optimizer framework.
impl<Q: Quantizer + Clone> NoopOptimizer<Q> {
    pub fn new() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}

/// This optimizer does nothing. It just copies the original segment to a new segment.
impl<Q: Quantizer + Clone> SegmentOptimizer<Q> for NoopOptimizer<Q> {
    fn optimize(&self, segment: &PendingSegment<Q>) -> Result<()> {
        let inner_segments = segment.inner_segments_names();
        let data_directory = segment.parent_directory();

        // Recursively copy everything from the data directory's inner segments to the new data directory.
        for inner_segment in inner_segments {
            let inner_segment_path = format!("{}/{}", data_directory, inner_segment);
            let pending_segment_path = format!("{}/{}", data_directory, segment.name());

            // Ignore errors if we can't create the directory. It might already exist.
            std::fs::create_dir_all(pending_segment_path.clone()).unwrap_or_default();

            // Copy the inner segment's contents to the new data directory
            let mut options = CopyOptions::default();
            options.content_only = true; // This ensures we copy only the contents, not the directory itself
            fs_extra::dir::copy(inner_segment_path, pending_segment_path, &options)?;
        }
        Ok(())
    }
}
