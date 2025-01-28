pub mod engine;
pub mod noop;

use anyhow::Result;
use quantization::quantization::Quantizer;

use crate::segment::pending_segment::PendingSegment;

pub trait SegmentOptimizer {
    fn optimize<Q: Quantizer>(&self, segment: &PendingSegment<Q>) -> Result<()>;
}
