use anyhow::Result;
use index::hnsw::builder::HnswBuilder;
use index::hnsw::writer::HnswWriter;
use index::vector::file::FileBackedAppendableVectorStorage;
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
        let vectors = Box::new(FileBackedAppendableVectorStorage::<u8>::new(
            vector_directory.clone(),
            self.config.max_memory_size,
            self.config.file_size,
            self.config.dimension / self.config.subvector_dimension,
        ));
        let mut hnsw_builder = HnswBuilder::new(
            self.config.max_num_neighbors,
            self.config.num_layers,
            self.config.ef_construction,
            Box::new(pq),
            vectors,
        );

        input.reset();
        while input.has_next() {
            let row = input.next();
            hnsw_builder.insert(row.id, row.data)?;
            if row.id % 10000 == 0 {
                debug!("Inserted {} rows", row.id);
            }
        }

        hnsw_builder.reindex()?;

        let hnsw_directory = format!("{}/hnsw", self.config.output_path);
        std::fs::create_dir_all(&hnsw_directory)?;

        info!("Start writing index");
        let hnsw_writer = HnswWriter::new(hnsw_directory);
        hnsw_writer.write(&mut hnsw_builder, true)?;

        // Cleanup tmp directory. It's ok to fail
        std::fs::remove_dir_all(&pg_temp_dir).unwrap_or_default();
        std::fs::remove_dir_all(&vector_directory).unwrap_or_default();
        Ok(())
    }
}
