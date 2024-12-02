use utils::distance::l2::L2DistanceCalculatorImpl;
use anyhow::Result;

pub trait Quantizer {
    /// Quantize a vector
    fn quantize(&self, value: &[f32]) -> Vec<u8>;

    /// Get the dimension of the quantized vector
    fn quantized_dimension(&self) -> usize;

    /// Get the original vector from the quantized vector.
    fn original_vector(&self, quantized_vector: &[u8]) -> Vec<f32>;

    /// Compute the distance between two quantized points
    fn distance(&self, query: &[u8], point: &[u8], implem: L2DistanceCalculatorImpl) -> f32;

    /// Read a quantizer
    fn read(dir: String) -> Result<Self> where Self: Sized;
}
