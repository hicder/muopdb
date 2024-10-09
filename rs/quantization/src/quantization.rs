pub trait Quantizer {
    fn quantize(&self, value: &[f32]) -> Vec<u8>;
}
