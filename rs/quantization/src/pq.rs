use std::fs::File;
use std::io::Write;
use std::path::Path;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use utils::l2::L2DistanceCalculatorImpl::{Scalar, SIMD};
use utils::l2::{L2DistanceCalculator, L2DistanceCalculatorImpl};
use utils::{DistanceCalculator, StreamingDistanceCalculator};

use crate::quantization::Quantizer;

const CODEBOOK_NAME: &str = "codebook";

pub struct ProductQuantizer {
    pub dimension: usize,
    pub subvector_dimension: usize,
    pub num_bits: u8,
    pub codebook: Vec<f32>,
    pub base_directory: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ProductQuantizerConfig {
    pub dimension: usize,
    pub subvector_dimension: usize,
    pub num_bits: u8,
}

impl ProductQuantizerConfig {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.dimension % self.subvector_dimension != 0 {
            return Err("Dimensions are not valid");
        }
        Ok(())
    }
}
pub struct ProductQuantizerReader {
    base_directory: String,
}

impl ProductQuantizerReader {
    pub fn new(base_directory: String) -> Self {
        Self { base_directory }
    }

    pub fn read(&self) -> Result<ProductQuantizer, &'static str> {
        let config_path = Path::new(&self.base_directory).join("product_quantizer_config.yaml");
        if !config_path.exists() {
            return Err("Config file does not exist");
        }

        if !config_path.is_file() {
            return Err("Config file is not a file");
        }

        let config_buffer = std::fs::read(config_path).unwrap();
        let config: ProductQuantizerConfig = serde_yaml::from_slice(&config_buffer).unwrap();

        match config.validate() {
            Ok(_) => {
                let pq = ProductQuantizer::load(config, &self.base_directory);
                return Ok(pq);
            }
            Err(e) => {
                return Err(e);
            }
        }
    }
}

pub struct ProductQuantizerWriter {
    base_directory: String,
}

impl ProductQuantizerWriter {
    pub fn new(base_directory: String) -> Self {
        Self { base_directory }
    }

    pub fn write(&self, quantizer: &ProductQuantizer) -> Result<()> {
        let config_path = Path::new(&self.base_directory).join("product_quantizer_config.yaml");
        if config_path.exists() {
            // Delete the file if exists
            std::fs::remove_file(config_path).unwrap();
        }

        // Write the codebook to a file
        let codebook_path = Path::new(&self.base_directory).join(&CODEBOOK_NAME);
        if codebook_path.exists() {
            // Delete the file if exists
            std::fs::remove_file(codebook_path).unwrap();
        }

        // Write codebook
        let codebook_buffer = quantizer.codebook_to_buffer();
        let mut codebook_file =
            File::create(Path::new(&self.base_directory).join(&CODEBOOK_NAME)).unwrap();
        codebook_file.write(&codebook_buffer)?;

        // Write config
        let mut config_file =
            File::create(Path::new(&self.base_directory).join("product_quantizer_config.yaml"))
                .unwrap();
        config_file.write(
            serde_yaml::to_string(&quantizer.config())
                .unwrap()
                .as_bytes(),
        )?;
        Ok(())
    }
}

impl ProductQuantizer {
    pub fn new(
        dimension: usize,
        subvector_dimension: usize,
        num_bits: u8,
        codebook: Vec<f32>,
        base_directory: String,
    ) -> Self {
        Self {
            dimension,
            subvector_dimension,
            num_bits,
            codebook,
            base_directory,
        }
    }

    pub fn load(config: ProductQuantizerConfig, base_directory: &str) -> Self {
        let codebook_path = Path::new(&base_directory).join("codebook");

        let codebook_buffer = std::fs::read(codebook_path).unwrap();
        let num_centroids = (1 << config.num_bits) as usize;
        let num_subvector = config.dimension / config.subvector_dimension;

        let mut offset = 0 as usize;
        let mut codebook = vec![];
        codebook.reserve_exact(num_subvector * num_centroids);

        for _ in 0..num_subvector * num_centroids {
            let val = f32::from_le_bytes(codebook_buffer[offset..offset + 4].try_into().unwrap());
            codebook.push(val);

            offset += 4;
        }

        Self {
            dimension: config.dimension,
            subvector_dimension: config.subvector_dimension,
            num_bits: config.num_bits,
            codebook,
            base_directory: base_directory.to_string(),
        }
    }

    pub fn codebook_to_buffer(&self) -> Vec<u8> {
        let mut codebook_buffer = vec![];
        codebook_buffer.reserve_exact(self.codebook.len() * 4);
        for codebook in self.codebook.iter() {
            let bytes = codebook.to_le_bytes();
            codebook_buffer.extend_from_slice(&bytes);
        }
        codebook_buffer
    }

    pub fn config(&self) -> ProductQuantizerConfig {
        ProductQuantizerConfig {
            dimension: self.dimension,
            subvector_dimension: self.subvector_dimension,
            num_bits: self.num_bits,
        }
    }
}

/// TODO(hicder): Make this faster
/// TODO(hicder): Support multiple distance type
impl Quantizer for ProductQuantizer {
    fn quantize(&self, value: &[f32]) -> Vec<u8> {
        let mut result = Vec::<u8>::with_capacity(self.dimension / self.subvector_dimension);
        let num_centroids = (1 << self.num_bits) as usize;
        let subspace_size_in_codebook = self.subvector_dimension * num_centroids;

        value
            .chunks_exact(self.subvector_dimension as usize)
            .enumerate()
            .for_each(|(subvector_idx, subvector)| {
                let subspace_offset = subvector_idx * subspace_size_in_codebook;
                let mut min_centroid_id = 0 as usize;
                let mut min_distance = std::f32::MAX;

                for i in 0..num_centroids {
                    let offset = subspace_offset + i * self.subvector_dimension;
                    let centroid = &self.codebook[offset..offset + self.subvector_dimension];
                    let distance = L2DistanceCalculator::new().calculate(&subvector, &centroid);
                    if distance < min_distance {
                        min_distance = distance;
                        min_centroid_id = i;
                    }
                }
                result.push(min_centroid_id as u8);
            });
        result
    }

    fn quantized_dimension(&self) -> usize {
        self.dimension / self.subvector_dimension
    }

    /// Get the original vector from the quantized vector.
    fn original_vector(&self, quantized_vector: &[u8]) -> Vec<f32> {
        let mut result = Vec::<f32>::with_capacity(self.dimension);
        let num_centroids = 1 << self.num_bits;
        quantized_vector
            .into_iter()
            .enumerate()
            .for_each(|(idx, quantized_value)| {
                let offset = idx * self.subvector_dimension * num_centroids;
                let centroid_offset =
                    offset + (*quantized_value as usize) * self.subvector_dimension;
                // TODO(hicder): This seems to be the hot path. SIMD this.
                for i in 0..self.subvector_dimension {
                    result.push(self.codebook[centroid_offset + i]);
                }
            });
        result
    }

    fn distance(&self, a: &[u8], b: &[u8], implem: L2DistanceCalculatorImpl) -> f32 {
        let num_centroids = 1 << self.num_bits;
        let mut distance_calculator = L2DistanceCalculator::new();
        let get_subvectors =
            |subvector_idx: usize, (a_quantized_value, b_quantized_value): (&u8, &u8)| {
                let offset = subvector_idx * self.subvector_dimension * num_centroids;
                let a_centroid_offset =
                    offset + (*a_quantized_value as usize) * self.subvector_dimension;
                let b_centroid_offset =
                    offset + (*b_quantized_value as usize) * self.subvector_dimension;

                let a_vec =
                    &self.codebook[a_centroid_offset..a_centroid_offset + self.subvector_dimension];
                let b_vec =
                    &self.codebook[b_centroid_offset..b_centroid_offset + self.subvector_dimension];

                (a_vec, b_vec)
            };

        match implem {
            Scalar | SIMD => {
                fn get_distance_fn(
                    implem: L2DistanceCalculatorImpl,
                ) -> impl Fn(&mut L2DistanceCalculator, &[f32], &[f32]) -> f32 {
                    move |calculator, a, b| {
                        if implem == Scalar {
                            calculator.calculate(a, b)
                        } else {
                            calculator.calculate_simd(a, b)
                        }
                    }
                }

                let calculate_distance = get_distance_fn(implem);

                a.iter()
                    .zip(b.iter())
                    .enumerate()
                    .map(|(idx, (a_val, b_val))| {
                        let (a_vec, b_vec) = get_subvectors(idx, (a_val, b_val));
                        let dist = calculate_distance(&mut distance_calculator, a_vec, b_vec);
                        dist * dist
                    })
                    .sum::<f32>()
                    .sqrt()
            }
            _ => {
                a.iter()
                    .zip(b.iter())
                    .enumerate()
                    .for_each(|(idx, (a_val, b_val))| {
                        let (a_vec, b_vec) = get_subvectors(idx, (a_val, b_val));
                        distance_calculator.stream(a_vec, b_vec);
                    });
                distance_calculator.finalize()
            }
        }
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
        // Create a temp directory
        let temp_dir = tempdir::TempDir::new("product_quantizer_test").unwrap();
        let base_directory = temp_dir.path().to_str().unwrap().to_string();

        let pq = ProductQuantizer::new(10, 2, 1, codebook, base_directory.clone());
        let value = vec![1.0, 1.0, 3.0, 3.0, 5.0, 5.0, 7.0, 7.0, 9.0, 9.0];
        let quantized_value = pq.quantize(&value);
        assert_eq!(quantized_value, vec![1, 1, 1, 1, 1]);

        // Write the codebook
        let writer = ProductQuantizerWriter {
            base_directory: base_directory.clone(),
        };
        writer.write(&pq).unwrap();

        // Read the quantizer
        let reader = ProductQuantizerReader {
            base_directory: base_directory.clone(),
        };

        let new_pq = match reader.read() {
            Ok(x) => x,
            Err(msg) => {
                panic!("{}", msg);
            }
        };
        assert_eq!(new_pq.dimension, 10);
        assert_eq!(new_pq.subvector_dimension, 2);
        assert_eq!(new_pq.num_bits, 1);
    }
}
