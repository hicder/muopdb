use std::{
    ops::Mul,
    simd::{f32x4, f32x8, num::SimdFloat},
};

use crate::DistanceCalculator;

pub struct L2DistanceCalculator {}

impl L2DistanceCalculator {
    pub fn new() -> Self {
        Self {}
    }

    #[allow(dead_code)]
    fn calculate_non_simd(&self, a: &[f32], b: &[f32]) -> f32 {
        let mut dist = 0.0;
        for i in 0..a.len() {
            dist += (a[i] - b[i]).powi(2);
        }
        dist.sqrt()
    }
}

impl DistanceCalculator for L2DistanceCalculator {
    fn calculate(&self, a: &[f32], b: &[f32]) -> f32 {
        let mut i = 0;
        let mut dist_8 = f32x8::splat(0.0);
        let mut dist_4 = f32x4::splat(0.0);
        let mut dist = 0.0;
        while i + 8 <= a.len() {
            let a = f32x8::from_array(a[i..i + 8].try_into().unwrap());
            let b = f32x8::from_array(b[i..i + 8].try_into().unwrap());
            let diff = a - b;
            dist_8 += diff.mul(diff);
            i += 8;
        }

        while i + 4 <= a.len() {
            let a = f32x4::from_array(a[i..i + 4].try_into().unwrap());
            let b = f32x4::from_array(b[i..i + 4].try_into().unwrap());
            let diff = a - b;
            dist_4 += diff.mul(diff);
            i += 4;
        }

        for j in i..a.len() {
            dist += (a[j] - b[j]).powi(2);
        }

        (dist_8.reduce_sum() + dist_4.reduce_sum() + dist).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use crate::test_utils::generate_random_vector;

    use super::*;
    use test::Bencher;
    extern crate test;

    #[test]
    fn test_l2() {
        // Create 2 random vectors of size 128
        let a = generate_random_vector(128);
        let b = generate_random_vector(128);

        let distance_calculator = L2DistanceCalculator::new();
        let distance_simd = distance_calculator.calculate(&a, &b);
        let distance_non_simd = distance_calculator.calculate_non_simd(&a, &b);

        let epsilon = 1e-4;
        assert!((distance_simd - distance_non_simd).abs() < epsilon);
    }

    #[bench]
    fn bench_simd(bencher: &mut Bencher) {
        let mut a = [0.0; 128];
        let mut b = [0.0; 128];
        for i in 0..128 {
            a[i] = i as f32;
            b[i] = (128 - i) as f32;
        }

        bencher.iter(|| L2DistanceCalculator::new().calculate(&a, &b));
    }

    #[bench]
    fn bench_non_simd(bencher: &mut Bencher) {
        let mut a = [0.0; 128];
        let mut b = [0.0; 128];
        for i in 0..128 {
            a[i] = i as f32;
            b[i] = (128 - i) as f32;
        }

        bencher.iter(|| L2DistanceCalculator::new().calculate_non_simd(&a, &b));
    }
}
