use num_traits::ToBytes;

use utils::distance::l2::L2DistanceCalculator;
use utils::distance::l2::L2DistanceCalculatorImpl::StreamingSIMD;
use utils::DistanceCalculator;
use crate::quantization::Quantizer;

pub trait VectorOps<Q: Quantizer> {
    fn process_vector(vector: &[f32], quantizer: &Q) -> Vec<Self> where Self: Sized;

    fn distance(vector: &[Self], other: &[Self], quantizer: &Q) -> f32 where Self: Sized;
}

impl<Q: Quantizer> VectorOps<Q> for u8 {
    fn process_vector(vector: &[f32], quantizer: &Q) -> Vec<u8> {
        quantizer.quantize(vector)
    }
    
    fn distance(vector: &[u8], other: &[u8], quantizer: &Q) -> f32 where Self: Sized {
        quantizer.distance(vector, other, StreamingSIMD)
    }
}

impl<Q: Quantizer> VectorOps<Q> for f32 {
    fn process_vector(vector: &[f32], _quantizer: &Q) -> Vec<f32> {
        vector.to_vec()
    }
    
    fn distance(vector: &[f32], other: &[f32], _quantizer: &Q) -> f32 where Self: Sized {
        L2DistanceCalculator::calculate(vector, other)
    }
}

pub trait VectorT<Q: Quantizer>: ToBytes + Clone + std::fmt::Debug + 'static + VectorOps<Q> {}

// Only u8 and f32
impl<Q: Quantizer> VectorT<Q> for u8 {}
impl<Q: Quantizer> VectorT<Q> for f32 {}
