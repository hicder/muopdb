use std::simd::{f32x4, f32x8};
use std::simd::num::SimdFloat;

use crate::{DistanceCalculator};

pub struct DotProductSimilarityCalculator {
    dist_simd_8: f32x8,
    dist_simd_4: f32x4,
    dist_simd_1: f32
}

#[derive(Debug, PartialEq, Clone)]
pub enum DotProductSimilarityCalculatorImpl {
    Scalar,
    SIMD,
    StreamingWithSIMD
}

impl DotProductSimilarityCalculator {
    pub fn new() -> Self {
        Self {
            dist_simd_8: f32x8::splat(0.0),
            dist_simd_4: f32x4::splat(0.0),
            dist_simd_1: 0.0
        }
    }

    pub fn calculate_scalar(&self, a: &[f32], b: &[f32]) -> f32 {
        // multiply corresponding elements in two vectors, then add them together
        a.iter()
            .zip(b.iter())
            .map(|(&x,&y)| x * y )
            .sum::<f32>()
    }

    fn multiply(&mut self, a: &[f32], b: &[f32]) {
        let mut i: usize = 0;
        let mut step: usize = self.dist_simd_8.len();

        while i + step <= a.len() {
            let a_slice = f32x8::from_slice(&a[i..]);
            let b_slice = f32x8::from_slice(&b[i..]);
            self.dist_simd_8 += a_slice * b_slice;
            i += step;
        }

        step = self.dist_simd_4.len();
        while i + step <= a.len() {
            let a_slice = f32x4::from_slice(&a[i..]);
            let b_slice = f32x4::from_slice(&b[i..]);
            self.dist_simd_4 += a_slice * b_slice;
            i += step;
        }

        for j in i..a.len() {
            self.dist_simd_1 += a[j] * b[j];
        }
    }

    fn accumulate(&self) -> f32 {
        self.dist_simd_8.reduce_sum() + self.dist_simd_4.reduce_sum() + self.dist_simd_1
    }

    pub fn calculate_simd(&mut self, a: &[f32], b: &[f32]) -> f32 {
        self.multiply(a, b);
        let res = self.accumulate();
        res
    }


}

impl DistanceCalculator for DotProductSimilarityCalculator {
    fn calculate(&mut self, a: &[f32], b: &[f32]) -> f32 {
        let num_elements = a.len();
        if num_elements < 32 {
            self.calculate_scalar(a, b)
        } else {
            self.calculate_simd(a, b)
        }
    }
}


// Tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::generate_random_vector;

    #[test]
    fn test_basic_dot_product_calculation() {
        // arrange
        let a = [0.0, 3.0, -5.0];
        let b = [-3.0, 9.0, 8.0];

        let mut distance_calculator = DotProductSimilarityCalculator::new();

        let known_product = 0.0 * -3.0 + 3.0 * 9.0 + -5.0 * 8.0;

        // act
        let distance_simd = distance_calculator.calculate_simd(&a, &b);
        let distance_scalar = distance_calculator.calculate_scalar(&a, &b);

        // assert
        assert_eq!(known_product, distance_simd);
        assert_eq!(known_product, distance_scalar);
    }

    #[test]
    fn test_dot_product_similarity_consistency() {
        // arrange
        let a = generate_random_vector(128);
        let b = generate_random_vector(128);

        let mut distance_calculator = DotProductSimilarityCalculator::new();
        let epsilon = 1e-5;

        // act
        let distance_simd = distance_calculator.calculate_simd(&a, &b);
        let distance_scalar = distance_calculator.calculate_scalar(&a,&b);

        // assert
        assert!((distance_simd - distance_scalar).abs() < epsilon);
    }
}

