
use crate::quantization::Quantizer;


pub struct ProductQuantizer {
    pub dimension: u32,
    pub subspace_dimension: u32,
    pub num_bits: u8,

    pub codebook: Vec<f32>,
}

impl Quantizer for ProductQuantizer {
    fn quantize(&self, value: &[f32]) -> Vec<u8> {
        todo!()
    }
}