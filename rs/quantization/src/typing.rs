use num_traits::ToBytes;
use utils::distance::l2::L2DistanceCalculatorImpl::StreamingSIMD;

use crate::quantization::Quantizer;

pub trait VectorOps<Q: Quantizer> {
    fn process_vector(vector: &[f32], quantizer: &Q) -> Vec<Q::QuantizedT>
    where
        Self: Sized;

    fn distance(vector: &[Q::QuantizedT], other: &[Q::QuantizedT], quantizer: &Q) -> f32
    where
        Self: Sized;
}

impl<Q: Quantizer> VectorOps<Q> for u8 {
    fn process_vector(vector: &[f32], quantizer: &Q) -> Vec<Q::QuantizedT> {
        quantizer.quantize(vector)
    }

    fn distance(vector: &[Q::QuantizedT], other: &[Q::QuantizedT], quantizer: &Q) -> f32
    where
        Self: Sized,
    {
        quantizer.distance(vector, other, StreamingSIMD)
    }
}

impl<Q: Quantizer> VectorOps<Q> for f32 {
    fn process_vector(vector: &[f32], quantizer: &Q) -> Vec<Q::QuantizedT> {
        quantizer.quantize(vector)
    }

    fn distance(vector: &[Q::QuantizedT], other: &[Q::QuantizedT], quantizer: &Q) -> f32
    where
        Self: Sized,
    {
        quantizer.distance(vector, other, StreamingSIMD)
    }
}

pub trait VectorT<Q: Quantizer>:
    ToBytes + Clone + std::fmt::Debug + 'static + VectorOps<Q>
{
}

// Only u8 and f32
impl<Q: Quantizer> VectorT<Q> for u8 {}
impl<Q: Quantizer> VectorT<Q> for f32 {}
