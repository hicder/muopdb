pub mod l2;
pub mod hdf5_reader;

pub trait DistanceCalculator {
    fn calculate(&self, a: &[f32], b: &[f32]) -> f32;
}
