pub mod hdf5_reader;
pub mod l2;
pub mod mem;
pub mod test_utils;

pub trait DistanceCalculator {
    fn calculate(&self, a: &[f32], b: &[f32]) -> f32;
}
