use anyhow::Result;
use compression::noc::noc::PlainEncoder;
use log::debug;
use quantization::noq::noq::{NoQuantizer, NoQuantizerWriter};

use super::builder::SpannBuilder;
use crate::hnsw::writer::HnswWriter;
use crate::ivf::writer::IvfWriter;

pub struct SpannWriter {
    base_directory: String,
}

impl SpannWriter {
    pub fn new(base_directory: String) -> Self {
        Self { base_directory }
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
        let quantizer_writer = NoQuantizerWriter::new(centroid_quantizer_directory);
        quantizer_writer.write(&spann_builder.centroid_builder.quantizer)?;

        // Write posting lists
        let ivf_directory = format!("{}/ivf", self.base_directory);
        std::fs::create_dir_all(&ivf_directory)?;
        let ivf_quantizer_directory = format!("{}/quantizer", ivf_directory);
        std::fs::create_dir_all(&ivf_quantizer_directory)?;

        let ivf_quantizer = NoQuantizer::new(index_writer_config.num_features);
        let ivf_quantizer_writer = NoQuantizerWriter::new(ivf_quantizer_directory);
        ivf_quantizer_writer.write(&ivf_quantizer)?;

        debug!("Writing IVF index");
        let ivf_writer = IvfWriter::<_, PlainEncoder>::new(ivf_directory, ivf_quantizer);
        ivf_writer.write(&mut spann_builder.ivf_builder, index_writer_config.reindex)?;
        spann_builder.ivf_builder.cleanup()?;
        debug!("Finish writing IVF index");

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

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
            max_iteration: 1000,
            batch_size: 4,
            num_clusters,
            num_data_points_for_clustering: num_vectors,
            max_clusters_per_vector: 1,
            distance_threshold: 0.1,
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
                .add(i as u64, &generate_random_vector(num_features))
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
