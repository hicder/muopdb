use std::ops::Mul;
use std::simd::num::SimdFloat;
use std::simd::{f32x16, f32x4, f32x8, LaneCount, Simd, SupportedLaneCount};

use strum::EnumIter;

use crate::{DistanceCalculator, StreamingDistanceCalculator};

#[derive(Debug, EnumIter, PartialEq, Clone)]
pub enum L2DistanceCalculatorImpl {
    Scalar,
    SIMD,
    StreamingWithSIMD,
}

pub struct L2DistanceCalculator {
    dist_simd_8: f32x8,
    dist_simd_4: f32x4,
    dist_simd_1: f32,
}

impl L2DistanceCalculator {
    pub fn new() -> Self {
        Self {
            dist_simd_8: f32x8::splat(0.0),
            dist_simd_4: f32x4::splat(0.0),
            dist_simd_1: 0.0,
        }
    }

    fn reset_distance_accumulators(&mut self) {
        self.dist_simd_8 = f32x8::splat(0.0);
        self.dist_simd_4 = f32x4::splat(0.0);
        self.dist_simd_1 = 0.0;
    }

    fn accumulate(&mut self, a: &[f32], b: &[f32]) {
        let mut i = 0;
        while i + 8 <= a.len() && i + 8 <= b.len() {
            let a_slice = f32x8::from_slice(&a[i..i + 8]);
            let b_slice = f32x8::from_slice(&b[i..i + 8]);
            let diff = a_slice - b_slice;
            self.dist_simd_8 += diff.mul(diff);
            i += 8;
        }
        while i + 4 <= a.len() && i + 4 <= b.len() {
            let a_slice = f32x4::from_slice(&a[i..i + 4]);
            let b_slice = f32x4::from_slice(&b[i..i + 4]);
            let diff = a_slice - b_slice;
            self.dist_simd_4 += diff.mul(diff);
            i += 4;
        }
        for j in i..a.len() {
            self.dist_simd_1 += (a[j] - b[j]).powi(2);
        }
    }

    fn reduce(&self) -> f32 {
        (self.dist_simd_8.reduce_sum() + self.dist_simd_4.reduce_sum() + self.dist_simd_1).sqrt()
    }

    pub fn calculate_simd(&mut self, a: &[f32], b: &[f32]) -> f32 {
        self.accumulate(a, b);
        let res = self.reduce();
        self.reset_distance_accumulators();
        res
    }

    pub fn calculate_scalar(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}

impl DistanceCalculator for L2DistanceCalculator {
    fn calculate(&mut self, a: &[f32], b: &[f32]) -> f32 {
        let num_elements = a.len();
        if num_elements < 32 {
            self.calculate_scalar(a, b)
        } else {
            self.calculate_simd(a, b)
        }
    }
}

impl StreamingDistanceCalculator for L2DistanceCalculator {
    fn stream(&mut self, a: &[f32], b: &[f32]) {
        self.accumulate(a, b);
    }

    fn finalize(&mut self) -> f32 {
        let res = self.reduce();
        self.reset_distance_accumulators();
        res
    }
}

/// Trait for calculating the squared distance between two vectors. An optimization for when the true
/// L2 distance is not needed.
pub trait CalculateSquared {
    fn calculate_squared(&self, a: &[f32], b: &[f32]) -> f32;
}

pub struct LaneConformingL2DistanceCalculator<const LANES: usize>
where
    LaneCount<LANES>: SupportedLaneCount, {}

impl<const LANES: usize> LaneConformingL2DistanceCalculator<LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    pub fn new() -> Self {
        Self {}
    }
}

impl<const LANES: usize> CalculateSquared for LaneConformingL2DistanceCalculator<LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    fn calculate_squared(&self, a: &[f32], b: &[f32]) -> f32 {
        let mut simd = Simd::<f32, LANES>::splat(0.0);
        let mut i = 0;
        while i + LANES <= a.len() && i + LANES <= b.len() {
            let a_slice = Simd::<f32, LANES>::from_slice(&a[i..i + LANES]);
            let b_slice = Simd::<f32, LANES>::from_slice(&b[i..i + LANES]);
            let diff = a_slice - b_slice;
            simd += diff.mul(diff);
            i += LANES;
        }
        simd.reduce_sum()
    }
}

pub struct NonStreamingL2DistanceCalculator {}

impl CalculateSquared for NonStreamingL2DistanceCalculator {
    fn calculate_squared(&self, a: &[f32], b: &[f32]) -> f32 {
        let mut simd_16 = f32x16::splat(0.0);
        let mut simd_8 = f32x8::splat(0.0);
        let mut simd_4 = f32x4::splat(0.0);
        let mut simd_1 = 0.0;
        let mut i = 0;
        while i + 16 <= a.len() && i + 16 <= b.len() {
            let a_slice = f32x16::from_slice(&a[i..i + 16]);
            let b_slice = f32x16::from_slice(&b[i..i + 16]);
            let diff = a_slice - b_slice;
            simd_16 += diff.mul(diff);
            i += 16;
        }

        while i + 8 <= a.len() && i + 8 <= b.len() {
            let a_slice = f32x8::from_slice(&a[i..i + 8]);
            let b_slice = f32x8::from_slice(&b[i..i + 8]);
            let diff = a_slice - b_slice;
            simd_8 += diff.mul(diff);
            i += 8;
        }

        while i + 4 <= a.len() && i + 4 <= b.len() {
            let a_slice = f32x4::from_slice(&a[i..i + 4]);
            let b_slice = f32x4::from_slice(&b[i..i + 4]);
            let diff = a_slice - b_slice;
            simd_4 += diff.mul(diff);
            i += 4;
        }

        for j in i..a.len() {
            simd_1 += (a[j] - b[j]).powi(2);
        }

        simd_16.reduce_sum() + simd_8.reduce_sum() + simd_4.reduce_sum() + simd_1.sqrt()
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

        let mut distance_calculator = L2DistanceCalculator::new();
        let epsilon = 1e-5;
        let distance_simd = distance_calculator.calculate_simd(&a, &b);
        let distance_scalar = distance_calculator.calculate_scalar(&a, &b);
        assert!((distance_simd - distance_scalar).abs() < epsilon);

        for i in (0..128).step_by(8) {
            let chunk_a = &a[i..i + 8];
            let chunk_b = &b[i..i + 8];
            distance_calculator.stream(chunk_a, chunk_b);
        }
        let distance_stream = distance_calculator.finalize();
        assert!((distance_stream - distance_scalar).abs() < epsilon);
    }
}
