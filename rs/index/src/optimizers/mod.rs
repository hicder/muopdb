pub mod engine;
pub mod merge;
pub mod noop;
pub mod vacuum;

use anyhow::Result;
use quantization::quantization::Quantizer;

use crate::segment::pending_segment::PendingSegment;

#[async_trait::async_trait]
pub trait SegmentOptimizer<Q: Quantizer + Clone + Send + Sync> {
    /// Optimizes the given pending segment.
    ///
    /// This method contains the core logic of the optimizer, processing the data within
    /// the `PendingSegment` to improve its structure, reduce size, or apply other
    /// optimizations (e.g., merging multiple segments, removing invalidated data).
    ///
    /// # Arguments
    ///
    /// * `segment` - The mutable pending segment to be optimized.
    ///
    /// # Returns
    ///
    /// A `Result` indicating success (`Ok(())`) or an error if the optimization fails.
    async fn optimize(&self, segment: &PendingSegment<Q>) -> Result<()>;
}
