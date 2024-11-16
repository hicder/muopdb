use anyhow::Result;
use kmeans::*;
use log::debug;

use crate::pq::{ProductQuantizer, ProductQuantizerConfig};

pub struct ProductQuantizerBuilderConfig {
    pub max_iteration: usize,
    pub batch_size: usize,
}

pub struct ProductQuantizerBuilder {
    pq_config: ProductQuantizerConfig,
    builder_config: ProductQuantizerBuilderConfig,
    dataset: Vec<Vec<f32>>,
}

impl ProductQuantizerBuilder {
    /// Create a new ProductQuantizerBuilder
    pub fn new(
        config: ProductQuantizerConfig,
        builder_config: ProductQuantizerBuilderConfig,
    ) -> Self {
        Self {
            pq_config: config,
            builder_config,
            dataset: Vec::new(),
        }
    }

    /// Add a new vector to the dataset for training
    pub fn add(&mut self, data: Vec<f32>) {
        self.dataset.push(data);
    }

    /// Train kmeans on the dataset, and returns the product quantizer
    pub fn build(&mut self, base_directory: String) -> Result<ProductQuantizer> {
        let num_subvector = self.pq_config.dimension / self.pq_config.subvector_dimension;
        let mut codebook = Vec::<f32>::with_capacity(
            num_subvector * self.pq_config.subvector_dimension * (1 << self.pq_config.num_bits),
        );

        for i in 0..num_subvector {
            let mut samples = vec![0.0f32; self.dataset.len() * self.pq_config.subvector_dimension];
            let mut idx = 0;
            for point in &self.dataset {
                let subvector = &point[i * self.pq_config.subvector_dimension
                    ..(i + 1) * self.pq_config.subvector_dimension];
                for f in subvector {
                    samples[idx] = *f;
                    idx += 1;
                }
            }
            let conf = KMeansConfig::build()
                .init_done(&|_| debug!("Initialization completed."))
                .iteration_done(&|s, nr, new_distsum| {
                    debug!(
                        "Iteration {} - Error: {:.2} -> {:.2} | Improvement: {:.2}",
                        nr,
                        s.distsum,
                        new_distsum,
                        s.distsum - new_distsum
                    )
                })
                .build();
            let kmean: KMeans<_, 8> = KMeans::new(
                samples,
                self.dataset.len(),
                self.pq_config.subvector_dimension,
            );
            let result = kmean.kmeans_minibatch(
                self.builder_config.batch_size,
                1 << self.pq_config.num_bits,
                self.builder_config.max_iteration,
                KMeans::init_random_sample,
                &conf,
            );
            result.centroids.iter().for_each(|x| codebook.push(*x));
            debug!("Error: {}", result.distsum);
        }
        ProductQuantizer::new(
            self.pq_config.dimension,
            self.pq_config.subvector_dimension,
            self.pq_config.num_bits,
            codebook,
            base_directory,
        )
    }
}

// Test
#[cfg(test)]
mod tests {
    use utils::distance::l2::L2DistanceCalculatorImpl::{Scalar, StreamingWithSIMD, SIMD};
    use utils::test_utils::generate_random_vector;

    use super::*;
    use crate::quantization::Quantizer;

    #[test]
    fn test_product_quantizer_builder() {
        env_logger::init();
        let base_directory = tempdir::TempDir::new("product_quantizer_test").unwrap();

        const DIMENSION: usize = 128;
        let mut pqb = ProductQuantizerBuilder::new(
            ProductQuantizerConfig {
                dimension: DIMENSION,
                subvector_dimension: 8,
                num_bits: 8,
            },
            ProductQuantizerBuilderConfig {
                max_iteration: 1000,
                batch_size: 4,
            },
        );
        // Generate 10000 vectors of f32, dimension 128
        for _ in 0..10000 {
            pqb.add(generate_random_vector(DIMENSION));
        }

        match pqb.build(base_directory.path().to_str().unwrap().to_string()) {
            Ok(_) => {
                assert!(true);
            }
            Err(_) => {
                assert!(false);
            }
        }
    }

    #[test]
    fn test_product_quantizer_distance() {
        const DIMENSION: usize = 128;
        let mut pqb = ProductQuantizerBuilder::new(
            ProductQuantizerConfig {
                dimension: DIMENSION,
                subvector_dimension: 8,
                num_bits: 8,
            },
            ProductQuantizerBuilderConfig {
                max_iteration: 1000,
                batch_size: 4,
            },
        );
        // Generate 10000 vectors of f32, dimension 128
        for _ in 0..10000 {
            pqb.add(generate_random_vector(DIMENSION));
        }

        let base_directory = tempdir::TempDir::new("product_quantizer_test").unwrap();
        let pq = pqb
            .build(base_directory.path().to_str().unwrap().to_string())
            .unwrap();
        let point = pq.quantize(&generate_random_vector(DIMENSION));
        let query = pq.quantize(&generate_random_vector(DIMENSION));
        let dist_scalar = pq.distance(&query, &point, Scalar);
        let dist_simd = pq.distance(&query, &point, SIMD);
        let dist_stream = pq.distance(&query, &point, StreamingWithSIMD);

        let epsilon = 1e-5;
        assert!((dist_simd - dist_scalar).abs() < epsilon);
        assert!((dist_stream - dist_scalar).abs() < epsilon);
    }
}
