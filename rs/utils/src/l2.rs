use std::ops::Mul;
use std::simd::num::SimdFloat;
use std::simd::{f32x4, f32x8};

use crate::DistanceCalculator;

pub struct L2DistanceCalculator {}

impl L2DistanceCalculator {
    pub fn new() -> Self {
        Self {}
    }

    pub fn calculate_simd(&self, a: &[f32], b: &[f32]) -> f32 {
        let mut i = 0;
        let mut dist_8 = f32x8::splat(0.0);
        let mut dist_4 = f32x4::splat(0.0);
        let mut dist = 0.0;
        while i + dist_8.len() <= a.len() {
            let a = f32x8::from_slice(&a[i..]);
            let b = f32x8::from_slice(&b[i..]);
            let diff = a - b;
            dist_8 += diff.mul(diff);
            i += dist_8.len();
        }
        while i + dist_4.len() <= a.len() {
            let a = f32x4::from_slice(&a[i..]);
            let b = f32x4::from_slice(&b[i..]);
            let diff = a - b;
            dist_4 += diff.mul(diff);
            i += dist_4.len();
        }
        for j in i..a.len() {
            dist += (a[j] - b[j]).powi(2);
        }

        (dist_8.reduce_sum() + dist_4.reduce_sum() + dist).sqrt()
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
    /// Compute L2 distance between two vectors
    fn calculate(&self, a: &[f32], b: &[f32]) -> f32 {
        self.calculate_scalar(a, b)
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

        let distance_calculator = L2DistanceCalculator::new();
        let distance_simd = distance_calculator.calculate_simd(&a, &b);
        let distance_scalar = distance_calculator.calculate_scalar(&a, &b);

        let epsilon = 1e-5;
        assert!((distance_simd - distance_scalar).abs() < epsilon);
    }
}
