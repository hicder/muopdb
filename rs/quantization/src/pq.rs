use crate::quantization::Quantizer;

pub struct ProductQuantizer {
    pub dimension: usize,
    pub subspace_dimension: usize,
    pub num_bits: u8,
    pub codebook: Vec<f32>,
}

impl ProductQuantizer {
    pub fn new(dimension: usize, subspace_dimension: usize, num_bits: u8, codebook: Vec<f32>) -> Self {
        Self {
            dimension,
            subspace_dimension,
            num_bits,
            codebook
        }
    }
}

/// Compute L2 distance between two vectors
/// TODO: Move this to a separate file
fn compute_l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f32>().sqrt()
}

/// TODO(hicder): Make this faster
/// TODO(hicder): Support multiple distance type
impl Quantizer for ProductQuantizer {
    fn quantize(&self, value: &[f32]) -> Vec<u8> {
        let mut result = Vec::<u8>::with_capacity(self.dimension/self.subspace_dimension);
        value.chunks_exact(self.subspace_dimension as usize).enumerate().for_each(
            |(subspace_idx, subspace_value)| {
                let num_centroids = (1 << self.num_bits) as usize;
                let subspace_size_in_codebook = self.subspace_dimension * num_centroids;
                let subspace_offset = subspace_idx * subspace_size_in_codebook;
                let mut min_centroid_id = 0 as usize;
                let mut min_distance = std::f32::MAX;

                for i in 0..num_centroids {
                    let offset = subspace_offset + i * self.subspace_dimension;
                    let centroid = &self.codebook[offset..offset + self.subspace_dimension];
                    let distance = compute_l2_distance(subspace_value, centroid);
                    if distance < min_distance {
                        min_distance = distance;
                        min_centroid_id = i;
                    }
                }
                result.push(min_centroid_id as u8);
            });
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_product_quantizer() {
        let mut codebook = vec![];
        for subvector_idx in 0..5 {
            for i in 0..(1 << 1) {
                let x = (subvector_idx * 2 + i) as f32;
                let y = (subvector_idx * 2 + i) as f32;
                codebook.push(x);
                codebook.push(y);
            }
        }
        let pq = ProductQuantizer::new(10, 2, 1, codebook);
        let value = vec![1.0, 1.0, 3.0, 3.0, 5.0, 5.0, 7.0, 7.0, 9.0, 9.0];
        let quantized_value = pq.quantize(&value);
        assert_eq!(quantized_value, vec![1, 1, 1, 1, 1]);
    }
}
