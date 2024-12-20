use crate::DistanceCalculator;
use std::{ops::AddAssign, simd::{num::SimdFloat, LaneCount, Simd, SupportedLaneCount}};

pub struct DotProductDistanceCalculator {}

impl DotProductDistanceCalculator {
    pub fn calculate_scalar(a: &[f32], b: &[f32]) -> f32 {
        let mut ret = 0.0;
        for i in 0..a.len() {
            ret += a[i] * b[i];
        }
        ret
    }
} 

impl DistanceCalculator for DotProductDistanceCalculator {
    fn calculate(a: &[f32], b: &[f32]) -> f32 {
        // test lanes 32 first
        let mut res = 0.0;
        let mut accumulator= Simd::<f32, 32>::splat(0.0);
        Self::accumulate_lanes::<32>(a, b, &mut accumulator);
        res += accumulator.reduce_sum();

        let a_vec = a.chunks_exact(32).remainder();
        let b_vec = b.chunks_exact(32).remainder();
        for i in 0..a_vec.len() {
            res += a_vec[i] * b_vec[i];
        } 
        res
    }

    fn accumulate_lanes<const LANES: usize>(
        a: &[f32],
        b: &[f32],
        accumulator: &mut Simd<f32, LANES>,
    ) where 
        LaneCount<LANES>: SupportedLaneCount, 
    {
        a.chunks_exact(LANES)
            .zip(b.chunks_exact(LANES))
            .for_each(|(a_chunk, b_chunk)| {
                let a_simd = Simd::<f32, LANES>::from_slice(a_chunk);
                let b_simd = Simd::<f32, LANES>::from_slice(b_chunk);
                accumulator.add_assign(a_simd * b_simd);
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::generate_random_vector;
    
    #[test]
    fn test_dot_product_distance_calculator() {
        let a = generate_random_vector(128);
        let b = generate_random_vector(128);
        let eps = 1e-5;
        let result = DotProductDistanceCalculator::calculate(&a, &b);
        let expected = DotProductDistanceCalculator::calculate_scalar(&a, &b);
        assert!((result - expected).abs() < eps);
    }
}