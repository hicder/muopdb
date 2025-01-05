use std::marker::PhantomData;
use std::simd::num::SimdFloat;
use std::simd::{LaneCount, Simd, SupportedLaneCount};

use crate::{CalculateSquared, DistanceCalculator};

/// Calculator where we know in advance that the dimension of vectors is a multiple of LANES.
/// This skips a bunch of checks and allows for a more efficient implementation.
pub struct LaneConformingDistanceCalculator<const LANES: usize, D: DistanceCalculator + Send + Sync>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    _marker: PhantomData<D>,
}

impl<const LANES: usize, D: DistanceCalculator + Send + Sync> CalculateSquared
    for LaneConformingDistanceCalculator<LANES, D>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    #[inline(always)]
    fn calculate_squared(a: &[f32], b: &[f32]) -> f32 {
        let mut simd = Simd::<f32, LANES>::splat(0.0);
        D::accumulate_lanes(a, b, &mut simd);
        D::outermost_op(simd.reduce_sum())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::dot_product::DotProductDistanceCalculator;
    use crate::distance::l2::L2DistanceCalculator;
    use crate::test_utils::generate_random_vector;

    #[test]
    fn test_calculate_l2_distance() {
        let a = generate_random_vector(16);
        let b = generate_random_vector(16);
        let eps = 1e-5;
        let conforming_result =
            LaneConformingDistanceCalculator::<4, L2DistanceCalculator>::calculate_squared(&a, &b);
        let l2_result = L2DistanceCalculator::calculate_squared(&a, &b);
        assert!((conforming_result - l2_result).abs() < eps)
    }

    #[test]
    fn test_calculate_dot_product_distance() {
        let a = generate_random_vector(16);
        let b = generate_random_vector(16);
        let eps = 1e-5;
        let conforming_result =
            LaneConformingDistanceCalculator::<4, DotProductDistanceCalculator>::calculate_squared(
                &a, &b,
            );
        let dot_product_result = DotProductDistanceCalculator::calculate_squared(&a, &b);
        assert!((conforming_result - dot_product_result).abs() < eps)
    }
}
