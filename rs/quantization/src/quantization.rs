use anyhow::Result;
use utils::distance::l2::L2DistanceCalculatorImpl;

use crate::typing::VectorT;

pub trait Quantizer: Send + Sync {
    type QuantizedT: VectorT<Self> + Send + Sync
    where
        Self: Sized;

    /// Quantize a vector
    fn quantize(&self, value: &[f32]) -> Vec<Self::QuantizedT>
    where
        Self: Sized;

    /// Get the dimension of the quantized vector
    fn quantized_dimension(&self) -> usize;

    /// Get the original vector from the quantized vector.
    fn original_vector(&self, quantized_vector: &[Self::QuantizedT]) -> Vec<f32>
    where
        Self: Sized;

    /// Compute the distance between two quantized points
    fn distance(
        &self,
        query: &[Self::QuantizedT],
        point: &[Self::QuantizedT],
        implem: L2DistanceCalculatorImpl,
    ) -> f32
    where
        Self: Sized;

    /// Read a quantizer
    fn read(dir: String) -> Result<Self>
    where
        Self: Sized;
}

pub trait WritableQuantizer: Quantizer {
    fn write_to_directory(&self, base_directory: &str) -> Result<()>;
}
