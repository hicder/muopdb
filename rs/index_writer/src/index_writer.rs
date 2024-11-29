use anyhow::Result;
use index::hnsw::builder::HnswBuilder;
use index::hnsw::writer::HnswWriter;
use log::{debug, info};
use quantization::pq::{ProductQuantizerConfig, ProductQuantizerWriter};
use quantization::pq_builder::{ProductQuantizerBuilder, ProductQuantizerBuilderConfig};
use rand::seq::SliceRandom;

use crate::config::{IndexWriterConfig, QuantizerType};
use crate::input::Input;

pub struct IndexWriter {
    config: IndexWriterConfig,
}

impl IndexWriter {
    pub fn new(config: IndexWriterConfig) -> Self {
        Self { config }
    }

    fn get_sorted_random_rows(num_rows: usize, num_random_rows: usize) -> Vec<u64> {
        let mut v = (0..num_rows).map(|x| x as u64).collect::<Vec<_>>();
        v.shuffle(&mut rand::thread_rng());
        let mut ret = v.into_iter().take(num_random_rows).collect::<Vec<u64>>();
        ret.sort();
        ret
    }

    // TODO(hicder): Support multiple inputs
    pub fn process(&mut self, input: &mut impl Input) -> Result<()> {
        info!("Start indexing");
        let pg_temp_dir = format!("{}/pq_tmp", self.config.output_path);
        std::fs::create_dir_all(&pg_temp_dir)?;

        // First, train the product quantizer
        let mut pq_builder = match self.config.quantizer_type {
            QuantizerType::ProductQuantizer => {
                let pq_config = ProductQuantizerConfig {
                    dimension: self.config.dimension,
                    subvector_dimension: self.config.subvector_dimension,
                    num_bits: self.config.num_bits,
                };
                let pq_builder_config = ProductQuantizerBuilderConfig {
                    max_iteration: self.config.max_iteration,
                    batch_size: self.config.batch_size,
                };
                ProductQuantizerBuilder::new(pq_config, pq_builder_config)
            }
        };

        info!("Start training product quantizer");
        let sorted_random_rows =
            Self::get_sorted_random_rows(input.num_rows(), self.config.num_training_rows);
        for row_idx in sorted_random_rows {
            input.skip_to(row_idx as usize);
            pq_builder.add(input.next().data.to_vec());
        }

        let pq = pq_builder.build(pg_temp_dir.clone())?;

        info!("Start writing product quantizer");
        let pq_directory = format!("{}/quantizer", self.config.output_path);
        std::fs::create_dir_all(&pq_directory)?;

        let pq_writer = ProductQuantizerWriter::new(pq_directory);
        pq_writer.write(&pq)?;

        info!("Start building index");
        let vector_directory = format!("{}/vectors", self.config.output_path);
        std::fs::create_dir_all(&vector_directory)?;

        let mut hnsw_builder = HnswBuilder::new(
            self.config.max_num_neighbors,
            self.config.num_layers,
            self.config.ef_construction,
            self.config.max_memory_size,
            self.config.file_size,
            self.config.dimension / self.config.subvector_dimension,
            Box::new(pq),
            vector_directory.clone(),
        );

        input.reset();
        while input.has_next() {
            let row = input.next();
            hnsw_builder.insert(row.id, row.data)?;
            if row.id % 10000 == 0 {
                debug!("Inserted {} rows", row.id);
            }
        }

        let hnsw_directory = format!("{}/hnsw", self.config.output_path);
        std::fs::create_dir_all(&hnsw_directory)?;

        info!("Start writing index");
        let hnsw_writer = HnswWriter::new(hnsw_directory);
        hnsw_writer.write(&mut hnsw_builder, self.config.reindex)?;

        // Cleanup tmp directory. It's ok to fail
        std::fs::remove_dir_all(&pg_temp_dir).unwrap_or_default();
        std::fs::remove_dir_all(&vector_directory).unwrap_or_default();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    use rand::Rng;

    use super::*;
    use crate::input::Row;

    // Mock Input implementation for testing
    struct MockInput {
        data: Vec<Vec<f32>>,
        current_index: usize,
    }

    impl MockInput {
        fn new(data: Vec<Vec<f32>>) -> Self {
            Self {
                data,
                current_index: 0,
            }
        }
    }

    impl Input for MockInput {
        fn num_rows(&self) -> usize {
            self.data.len()
        }

        fn skip_to(&mut self, index: usize) {
            self.current_index = index;
        }

        fn next(&mut self) -> Row {
            let row = Row {
                id: self.current_index as u64,
                data: &self.data[self.current_index],
            };
            self.current_index += 1;
            row
        }

        fn has_next(&self) -> bool {
            self.current_index < self.data.len()
        }

        fn reset(&mut self) {
            self.current_index = 0;
        }
    }

    #[test]
    fn test_get_sorted_random_rows() {
        let num_rows = 100;
        let num_random_rows = 50;
        let result = IndexWriter::get_sorted_random_rows(num_rows, num_random_rows);
        assert_eq!(result.len(), num_random_rows);
        for i in 1..result.len() {
            assert!(result[i - 1] <= result[i]);
        }
    }

    #[test]
    fn test_index_writer_process() {
        // Setup test data
        let mut rng = rand::thread_rng();
        let dimension = 10;
        let num_rows = 100;
        let data: Vec<Vec<f32>> = (0..num_rows)
            .map(|_| (0..dimension).map(|_| rng.gen::<f32>()).collect())
            .collect();

        let mut mock_input = MockInput::new(data);

        // Create a temporary directory for output
        let temp_dir = Path::new("test_output");
        if temp_dir.exists() {
            fs::remove_dir_all(temp_dir).unwrap();
        }
        fs::create_dir_all(temp_dir).unwrap();

        // Configure IndexWriter
        let config = IndexWriterConfig {
            output_path: temp_dir.to_str().unwrap().to_string(),
            dimension,
            subvector_dimension: 2,
            num_bits: 2,
            max_iteration: 10,
            batch_size: 10,
            num_training_rows: 50,
            max_num_neighbors: 10,
            num_layers: 2,
            ef_construction: 100,
            max_memory_size: 1024 * 1024 * 1024, // 1 GB
            file_size: 1024 * 1024 * 1024,       // 1 GB
            quantizer_type: QuantizerType::ProductQuantizer,
            reindex: false,
        };

        let mut index_writer = IndexWriter::new(config);

        // Process the input
        index_writer.process(&mut mock_input).unwrap();

        // Check if output directories and files exist
        let pq_directory_path = format!("{}/quantizer", temp_dir.to_str().unwrap());
        let pq_directory = Path::new(&pq_directory_path);
        let hnsw_directory_path = format!("{}/hnsw", temp_dir.to_str().unwrap());
        let hnsw_directory = Path::new(&hnsw_directory_path);
        assert!(pq_directory.exists());
        assert!(hnsw_directory.exists());

        // Cleanup
        fs::remove_dir_all(temp_dir).unwrap();
    }
}
