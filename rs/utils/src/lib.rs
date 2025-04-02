#![feature(portable_simd)]
#![feature(btree_cursors)]
#![feature(new_range_api)]

use std::simd::{LaneCount, Simd, SupportedLaneCount};
pub mod distance;
pub mod io;
pub mod kmeans_builder;
pub mod mem;
pub mod on_disk_ordered_map;
pub mod test_utils;

pub trait DistanceCalculator {
    /// Compute distance between two vectors.
    fn calculate(a: &[f32], b: &[f32]) -> f32;

    /// Compute distance between two vectors using SIMD.
    fn accumulate_lanes<const LANES: usize>(
        a: &[f32],
        b: &[f32],
        accumulator: &mut Simd<f32, LANES>,
    ) where
        LaneCount<LANES>: SupportedLaneCount;

    fn accumulate_scalar(a: &[f32], b: &[f32]) -> f32;

    /*
     * The outermost operator of the distance function,
     * to be used with accumulate_lanes for lane conforming code.
     */
    fn outermost_op(x: f32) -> f32;
}

pub trait CalculateSquared {
    fn calculate_squared(a: &[f32], b: &[f32]) -> f32;
}

#[inline(always)]
pub fn ceil_div(a: usize, b: usize) -> usize {
    a.div_ceil(b)
}
