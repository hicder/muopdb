use std::sync::Arc;

use anyhow::Result;
use quantization::quantization::Quantizer;
use utils::file_io::env::Env;

use crate::hnsw::block_based::index::BlockBasedHnsw;

pub struct HnswReader {
    base_directory: String,
    index_offset: usize,
    vector_offset: usize,
}

impl HnswReader {
    pub fn new(base_directory: String) -> Self {
        Self {
            base_directory,
            index_offset: 0,
            vector_offset: 0,
        }
    }

    pub fn new_with_offset(
        base_directory: String,
        index_offset: usize,
        vector_offset: usize,
    ) -> Self {
        Self {
            base_directory,
            index_offset,
            vector_offset,
        }
    }

    pub async fn read<Q: Quantizer>(&self, env: Arc<Box<dyn Env>>) -> Result<BlockBasedHnsw<Q>>
    where
        Q::QuantizedT: Send + Sync,
    {
        BlockBasedHnsw::new_with_offsets(
            env,
            self.base_directory.clone(),
            self.index_offset,
            self.vector_offset,
        )
        .await
    }
}

// Test
#[cfg(test)]
mod tests {
    use std::fs;

    use quantization::noq::noq::NoQuantizer;
    use quantization::pq::pq::{ProductQuantizer, ProductQuantizerConfig};
    use quantization::pq::pq_builder::{ProductQuantizerBuilder, ProductQuantizerBuilderConfig};
    use quantization::quantization::WritableQuantizer;
    use utils::distance::l2::L2DistanceCalculator;
    use utils::file_io::env::{DefaultEnv, EnvConfig, FileType};
    use utils::test_utils::generate_random_vector;

    use super::*;
    use crate::hnsw::builder::HnswBuilder;
    use crate::hnsw::writer::HnswWriter;

    fn create_env() -> Arc<Box<dyn Env>> {
        let config = EnvConfig {
            file_type: FileType::CachedStandard,
            ..EnvConfig::default()
        };
        Arc::new(Box::new(DefaultEnv::new(config)))
    }

    #[tokio::test]
    async fn test_read_header() {
        // Generate 10000 vectors of f32, dimension 128
        let datapoints: Vec<Vec<f32>> = (0..10000).map(|_| generate_random_vector(128)).collect();

        // Create a temporary directory
        let temp_dir = tempdir::TempDir::new("product_quantizer_test").unwrap();
        let base_directory = temp_dir.path().to_str().unwrap().to_string();
        let pq_dir = format!("{}/quantizer", base_directory);
        fs::create_dir_all(pq_dir.clone()).unwrap();
        let pq_config = ProductQuantizerConfig {
            dimension: 128,
            subvector_dimension: 8,
            num_bits: 8,
        };

        let pq_builder_config = ProductQuantizerBuilderConfig {
            max_iteration: 1000,
            batch_size: 4,
        };

        // Train a product quantizer
        let mut pq_builder = ProductQuantizerBuilder::new(pq_config, pq_builder_config);

        for datapoint in datapoints.iter().take(1000) {
            pq_builder.add(datapoint.clone());
        }
        let pq = pq_builder.build(base_directory.clone()).unwrap();
        assert!(pq.write_to_directory(&pq_dir).is_ok());

        // Create a HNSW Builder
        let vector_dir = format!("{}/vectors", base_directory);
        fs::create_dir_all(vector_dir.clone()).unwrap();
        let mut hnsw_builder = HnswBuilder::<ProductQuantizer<L2DistanceCalculator>>::new(
            10, 128, 20, 1024, 4096, 16, pq, vector_dir,
        );
        for (i, datapoint) in datapoints.iter().enumerate() {
            hnsw_builder.insert(i as u128, datapoint).unwrap();
        }

        let hnsw_dir = format!("{}/hnsw", base_directory);
        fs::create_dir_all(hnsw_dir.clone()).unwrap();
        let writer = HnswWriter::new(hnsw_dir);
        assert!(writer.write(&mut hnsw_builder, false).is_ok());

        // Read from file
        let env = create_env();
        let reader = HnswReader::new(base_directory.clone());
        let hnsw = reader
            .read::<ProductQuantizer<L2DistanceCalculator>>(env)
            .await
            .unwrap();
        assert_eq!(16, hnsw.get_header().quantized_dimension);
    }

    #[tokio::test]
    async fn test_read_no_op_quantizer() {
        let temp_dir = tempdir::TempDir::new("product_quantizer_test").unwrap();
        let base_directory = temp_dir.path().to_str().unwrap().to_string();
        let vector_dir = format!("{}/vectors", base_directory);
        fs::create_dir_all(vector_dir.clone()).unwrap();
        let datapoints: Vec<Vec<f32>> = (0..10000).map(|_| generate_random_vector(128)).collect();

        // quantizer
        let quantizer = NoQuantizer::<L2DistanceCalculator>::new(128);
        let quantizer_dir = format!("{}/quantizer", base_directory);
        fs::create_dir_all(quantizer_dir.clone()).unwrap();
        assert!(quantizer.write_to_directory(&quantizer_dir).is_ok());

        let mut hnsw_builder =
            HnswBuilder::new(10, 128, 20, 1024, 4096, 128, quantizer, vector_dir);
        for (i, datapoint) in datapoints.iter().enumerate() {
            hnsw_builder.insert(i as u128, datapoint).unwrap();
        }

        let hnsw_dir = format!("{}/hnsw", base_directory);
        fs::create_dir_all(hnsw_dir.clone()).unwrap();
        let writer = HnswWriter::new(hnsw_dir);
        assert!(writer.write(&mut hnsw_builder, false).is_ok());

        // Read from file
        let env = create_env();
        let reader = HnswReader::new(base_directory.clone());
        let hnsw = reader
            .read::<NoQuantizer<L2DistanceCalculator>>(env)
            .await
            .unwrap();
        assert_eq!(128, hnsw.get_header().quantized_dimension);
    }
}
