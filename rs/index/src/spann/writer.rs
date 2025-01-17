use anyhow::Result;
use compression::noc::noc::PlainEncoder;
use config::enums::QuantizerType;
use log::debug;
use quantization::noq::noq::NoQuantizer;
use quantization::pq::pq::ProductQuantizerConfig;
use quantization::pq::pq_builder::{ProductQuantizerBuilder, ProductQuantizerBuilderConfig};
use quantization::quantization::WritableQuantizer;
use rand::prelude::SliceRandom;
use utils::distance::l2::L2DistanceCalculator;

use super::builder::SpannBuilder;
use crate::hnsw::writer::HnswWriter;
use crate::ivf::builder::IvfBuilder;
use crate::ivf::writer::IvfWriter;
use crate::spann::builder::SpannBuilderConfig;

pub struct SpannWriter {
    base_directory: String,
}

impl SpannWriter {
    pub fn new(base_directory: String) -> Self {
        Self { base_directory }
    }

    fn get_sorted_random_rows(num_rows: usize, num_random_rows: usize) -> Vec<u64> {
        let mut v = (0..num_rows).map(|x| x as u64).collect::<Vec<_>>();
        v.shuffle(&mut rand::thread_rng());
        let mut ret = v.into_iter().take(num_random_rows).collect::<Vec<u64>>();
        ret.sort();
        ret
    }

    pub fn write_ivf_pq(
        ivf_directory: &str,
        index_writer_config: &SpannBuilderConfig,
        ivf_builder: &mut IvfBuilder<L2DistanceCalculator>,
    ) -> Result<()> {
        // Create and train product quantizer
        let pq_config = ProductQuantizerConfig {
            dimension: index_writer_config.num_features,
            subvector_dimension: index_writer_config.subvector_dimension,
            num_bits: index_writer_config.num_bits as u8,
        };

        let pq_builder_config = ProductQuantizerBuilderConfig {
            max_iteration: index_writer_config.max_iteration,
            batch_size: index_writer_config.batch_size,
        };

        let mut pq_builder =
            ProductQuantizerBuilder::<L2DistanceCalculator>::new(pq_config, pq_builder_config);

        debug!("Start training product quantizer");
        let sorted_random_rows = Self::get_sorted_random_rows(
            ivf_builder.vectors().borrow().len(),
            index_writer_config.num_training_rows,
        );

        for row_idx in sorted_random_rows {
            let vector = ivf_builder.vectors().borrow().get(row_idx as u32)?.to_vec();
            pq_builder.add(vector);
        }

        let pq = pq_builder.build(format!("{}/pq_tmp", ivf_directory))?;

        let ivf_quantizer_directory = format!("{}/quantizer", ivf_directory);
        std::fs::create_dir_all(&ivf_quantizer_directory)?;
        pq.write_to_directory(&ivf_quantizer_directory)?;

        debug!("Writing IVF index");
        let ivf_writer =
            IvfWriter::<_, PlainEncoder, L2DistanceCalculator>::new(ivf_directory.to_string(), pq);
        ivf_writer.write(ivf_builder, index_writer_config.reindex)?;
        ivf_builder.cleanup()?;
        debug!("Finish writing IVF index");
        Ok(())
    }

    pub fn write_ivf_noq(
        ivf_directory: &str,
        index_writer_config: &SpannBuilderConfig,
        ivf_builder: &mut IvfBuilder<L2DistanceCalculator>,
    ) -> Result<()> {
        let ivf_quantizer_directory = format!("{}/quantizer", ivf_directory);
        std::fs::create_dir_all(&ivf_quantizer_directory)?;
        let ivf_quantizer =
            NoQuantizer::<L2DistanceCalculator>::new(index_writer_config.num_features);
        ivf_quantizer.write_to_directory(&ivf_quantizer_directory)?;

        debug!("Writing IVF index");
        let ivf_writer = IvfWriter::<_, PlainEncoder, L2DistanceCalculator>::new(
            ivf_directory.to_string(),
            ivf_quantizer,
        );
        ivf_writer.write(ivf_builder, index_writer_config.reindex)?;
        ivf_builder.cleanup()?;
        debug!("Finish writing IVF index");
        Ok(())
    }

    pub fn write(&self, spann_builder: &mut SpannBuilder) -> Result<()> {
        let index_writer_config = &spann_builder.config;

        let centroid_directory = format!("{}/centroids", self.base_directory);
        std::fs::create_dir_all(&centroid_directory)?;

        let centroid_quantizer_directory = format!("{}/quantizer", centroid_directory);
        std::fs::create_dir_all(&centroid_quantizer_directory)?;

        let hnsw_directory = format!("{}/hnsw", centroid_directory);
        std::fs::create_dir_all(&hnsw_directory)?;

        debug!("Writing centroids");
        let hnsw_writer = HnswWriter::new(hnsw_directory);
        hnsw_writer.write(
            &mut spann_builder.centroid_builder,
            index_writer_config.reindex,
        )?;
        debug!("Finish writing centroids");

        // Write the quantizer to disk, even though it's no quantizer
        spann_builder
            .centroid_builder
            .quantizer
            .write_to_directory(&centroid_quantizer_directory)?;

        // Write posting lists
        let ivf_directory = format!("{}/ivf", self.base_directory);
        std::fs::create_dir_all(&ivf_directory)?;

        match index_writer_config.quantizer_type {
            QuantizerType::ProductQuantizer => {
                Self::write_ivf_pq(
                    &ivf_directory,
                    &index_writer_config,
                    &mut spann_builder.ivf_builder,
                )?;
            }
            QuantizerType::NoQuantizer => {
                Self::write_ivf_noq(
                    &ivf_directory,
                    &index_writer_config,
                    &mut spann_builder.ivf_builder,
                )?;
            }
        };

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use config::enums::{IntSeqEncodingType, QuantizerType};
    use tempdir::TempDir;
    use utils::test_utils::generate_random_vector;

    use super::*;
    use crate::spann::builder::SpannBuilderConfig;

    #[test]
    fn test_write() {
        let temp_dir = TempDir::new("test_write").unwrap();

        let base_directory = temp_dir.path().to_str().unwrap().to_string();
        let num_clusters = 10;
        let num_vectors = 1000;
        let num_features = 4;
        let file_size = 4096;
        let balance_factor = 0.0;
        let max_posting_list_size = usize::MAX;
        let mut builder = SpannBuilder::new(SpannBuilderConfig {
            max_neighbors: 10,
            max_layers: 2,
            ef_construction: 100,
            vector_storage_memory_size: 1024,
            vector_storage_file_size: file_size,
            num_features,
            subvector_dimension: 8,
            num_bits: 8,
            num_training_rows: 50,
            quantizer_type: QuantizerType::NoQuantizer,
            max_iteration: 1000,
            batch_size: 4,
            num_clusters,
            num_data_points_for_clustering: num_vectors,
            max_clusters_per_vector: 1,
            distance_threshold: 0.1,
            posting_list_encoding_type: IntSeqEncodingType::PlainEncoding,
            base_directory: base_directory.clone(),
            memory_size: 1024,
            file_size,
            tolerance: balance_factor,
            max_posting_list_size,
            reindex: false,
        })
        .unwrap();

        // Generate 1000 vectors of f32, dimension 4
        for i in 0..num_vectors {
            builder
                .add(i as u128, &generate_random_vector(num_features))
                .unwrap();
        }
        builder.build().unwrap();

        let spann_writer = SpannWriter::new(base_directory.clone());
        spann_writer.write(&mut builder).unwrap();

        // Check if output directories and files exist
        let centroids_directory_path = format!("{}/centroids/hnsw", base_directory);
        let centroids_directory = PathBuf::from(&centroids_directory_path);
        let hnsw_vector_storage_path =
            format!("{}/vector_storage", centroids_directory.to_str().unwrap());
        let hnsw_index_path = format!("{}/index", centroids_directory.to_str().unwrap());

        assert!(PathBuf::from(&centroids_directory_path).exists());
        assert!(PathBuf::from(&hnsw_vector_storage_path).exists());
        assert!(PathBuf::from(&hnsw_index_path).exists());

        let ivf_directory_path = format!("{}/ivf", base_directory);
        let ivf_directory = PathBuf::from(&ivf_directory_path);
        let ivf_vector_storage_path = format!("{}/vectors", ivf_directory.to_str().unwrap());
        let ivf_index_path = format!("{}/index", ivf_directory.to_str().unwrap());
        assert!(PathBuf::from(&ivf_directory_path).exists());
        assert!(PathBuf::from(&ivf_vector_storage_path).exists());
        assert!(PathBuf::from(&ivf_index_path).exists());
    }
}
