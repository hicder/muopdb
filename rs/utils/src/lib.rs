#![feature(portable_simd)]
pub mod hdf5_reader;
pub mod io;
pub mod l2;
pub mod mem;
pub mod test_utils;

pub trait DistanceCalculator {
    /// Compute distance between two vectors.
    fn calculate(&self, a: &[f32], b: &[f32]) -> f32;
}

pub trait StreamingDistanceCalculator {
    fn stream(&mut self, a: &[f32], b: &[f32]);
    fn finalize(&mut self) -> f32;
}
