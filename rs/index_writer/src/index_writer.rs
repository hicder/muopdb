use anyhow::Result;
use index::hnsw::builder::HnswBuilder;
use index::hnsw::writer::HnswWriter;
use index::vector::file::FileBackedVectorStorage;
use log::info;
use quantization::pq::{ProductQuantizerConfig, ProductQuantizerWriter};
use quantization::pq_builder::{ProductQuantizerBuilder, ProductQuantizerBuilderConfig};

use crate::config::{IndexWriterConfig, QuantizerType};
use crate::input::Input;

pub struct IndexWriter {
    config: IndexWriterConfig,
}

impl IndexWriter {
    pub fn new(config: IndexWriterConfig) -> Self {
        Self { config }
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

        // TODO(hicder): Sample instead of getting the first rows
        let mut num_added = 0;
        while input.has_next() && num_added < self.config.num_training_rows {
            let row = input.next();
            pq_builder.add(row.data.to_vec());
            num_added += 1;
        }
        let pq = pq_builder.build(pg_temp_dir)?;

        info!("Start writing product quantizer");
        let pq_directory = format!("{}/quantizer", self.config.output_path);
        std::fs::create_dir_all(&pq_directory)?;

        let pq_writer = ProductQuantizerWriter::new(pq_directory);
        pq_writer.write(&pq)?;

        info!("Start building index");
        let vector_directory = format!("{}/vectors", self.config.output_path);
        std::fs::create_dir_all(&vector_directory)?;
        let vectors = Box::new(FileBackedVectorStorage::<u8>::new(
            vector_directory,
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
        }

        hnsw_builder.reindex()?;

        let hnsw_directory = format!("{}/hnsw", self.config.output_path);
        std::fs::create_dir_all(&hnsw_directory)?;

        info!("Start writing index");
        let hnsw_writer = HnswWriter::new(hnsw_directory);
        hnsw_writer.write(&mut hnsw_builder, true)?;
        Ok(())
    }
}
