use std::ops::Mul;
use std::simd::num::SimdFloat;
use std::simd::{f32x16, f32x4, f32x8, LaneCount, Simd, SupportedLaneCount};

use strum::EnumIter;

use crate::{CalculateSquared, DistanceCalculator};

#[derive(Debug, EnumIter, PartialEq, Clone)]
pub enum L2DistanceCalculatorImpl {
    Scalar,
    SIMD,
    StreamingSIMD,
}

#[derive(Debug, Clone)]
pub struct L2DistanceCalculator {}

impl L2DistanceCalculator {
    #[inline(always)]
    pub fn calculate_scalar(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}

impl CalculateSquared for L2DistanceCalculator {
    #[inline(always)]
    fn calculate_squared(a: &[f32], b: &[f32]) -> f32 {
        let mut a_vec = a;
        let mut b_vec = b;
        let mut ret = 0.0;

        if a_vec.len() / 16 > 0 {
            let mut sum_16 = f32x16::splat(0.0);
            Self::accumulate_lanes::<16>(a_vec, b_vec, &mut sum_16);
            a_vec = a_vec.chunks_exact(16).remainder();
            b_vec = b_vec.chunks_exact(16).remainder();
            ret += sum_16.reduce_sum();
        }

        if a_vec.len() / 8 > 0 {
            let mut sum_8 = f32x8::splat(0.0);
            Self::accumulate_lanes::<8>(a_vec, b_vec, &mut sum_8);
            a_vec = a_vec.chunks_exact(8).remainder();
            b_vec = b_vec.chunks_exact(8).remainder();
            ret += sum_8.reduce_sum();
        }

        if a_vec.len() / 4 > 0 {
            let mut sum_4 = f32x4::splat(0.0);
            Self::accumulate_lanes::<4>(a_vec, b_vec, &mut sum_4);
            a_vec = a_vec.chunks_exact(4).remainder();
            b_vec = b_vec.chunks_exact(4).remainder();
            ret += sum_4.reduce_sum();
        }

        if !a_vec.is_empty() {
            for i in 0..a_vec.len() {
                ret += (a_vec[i] - b_vec[i]).powi(2);
            }
        }
        ret
    }
}

impl DistanceCalculator for L2DistanceCalculator {
    #[inline(always)]
    fn calculate(a: &[f32], b: &[f32]) -> f32 {
        Self::calculate_squared(a, b).sqrt()
    }

    #[inline(always)]
    fn accumulate_lanes<const LANES: usize>(a: &[f32], b: &[f32], acc: &mut Simd<f32, LANES>)
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        a.chunks_exact(LANES)
            .zip(b.chunks_exact(LANES))
            .for_each(|(a, b)| {
                let a_slice = Simd::<f32, LANES>::from_slice(a);
                let b_slice = Simd::<f32, LANES>::from_slice(b);
                let diff = a_slice - b_slice;
                *acc += diff.mul(diff);
            });
    }

    #[inline(always)]
    fn accumulate_scalar(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).powi(2)).sum()
    }

    #[inline(always)]
    fn outermost_op(x: f32) -> f32 {
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::generate_random_vector;

    #[test]
    fn test_l2_impls() {
        // Create 2 random vectors of size 128
        let a = generate_random_vector(128);
        let b = generate_random_vector(128);

        let epsilon = 1e-5;
        let distance_simd = L2DistanceCalculator::calculate(&a, &b);
        let distance_scalar = L2DistanceCalculator::calculate_scalar(&a, &b);
        assert!((distance_simd - distance_scalar).abs() < epsilon);
    }

    #[test]
    fn test_accumulate_scalar() {
        // Create 2 random vectors of size 128
        let a = generate_random_vector(128);
        let b = generate_random_vector(128);

        let epsilon = 1e-5;
        let distance_scalar = L2DistanceCalculator::calculate_scalar(&a, &b);
        let accumulate_scalar = L2DistanceCalculator::accumulate_scalar(&a, &b);
        assert!((distance_scalar - accumulate_scalar.sqrt()) < epsilon)
    }
}
