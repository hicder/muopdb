pub mod engine;
pub mod merge;
pub mod noop;
pub mod vacuum;

use anyhow::Result;
use quantization::quantization::Quantizer;

use crate::segment::pending_segment::PendingSegment;

pub trait SegmentOptimizer<Q: Quantizer + Clone + Send + Sync> {
    fn optimize(&self, segment: &PendingSegment<Q>) -> Result<()>;
}
