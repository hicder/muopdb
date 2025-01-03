use anyhow::Result;
use quantization::quantization::Quantizer;

use crate::ivf::index::Ivf;
use crate::posting_list::combined_file::FixedIndexFile;
use crate::vector::fixed_file::FixedFileVectorStorage;

pub struct IvfReader {
    base_directory: String,
}

impl IvfReader {
    pub fn new(base_directory: String) -> Self {
        Self { base_directory }
    }

    pub fn read<Q: Quantizer>(&self) -> Result<Ivf<Q>> {
        let index_storage = FixedIndexFile::new(format!("{}/index", self.base_directory))?;

        let vector_storage_path = format!("{}/vectors", self.base_directory);
        let vector_storage = FixedFileVectorStorage::<Q::QuantizedT>::new(
            vector_storage_path,
            index_storage.header().quantized_dimension as usize,
        )?;

        let num_clusters = index_storage.header().num_clusters as usize;

        // Read quantizer
        let quantizer_directory = format!("{}/quantizer", self.base_directory);
        let quantizer = Q::read(quantizer_directory).unwrap();

        Ok(Ivf::new(
            vector_storage,
            index_storage,
            num_clusters,
            quantizer,
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use quantization::noq::noq::{NoQuantizer, NoQuantizerWriter};
    use tempdir::TempDir;
    use utils::distance::l2::L2DistanceCalculator;
    use utils::test_utils::generate_random_vector;

    use super::*;
    use crate::ivf::builder::{IvfBuilder, IvfBuilderConfig};
    use crate::ivf::writer::IvfWriter;
    use crate::posting_list::combined_file::Version;
    use crate::utils::SearchContext;

    #[test]
    fn test_ivf_reader_read() {
        let temp_dir =
            TempDir::new("test_ivf_reader_read").expect("Failed to create temporary directory");
        let base_directory = temp_dir
            .path()
            .to_str()
            .expect("Failed to convert temporary directory path to string")
            .to_string();
        let num_clusters = 10;
        let num_vectors = 1000;
        let num_features = 4;
        let file_size = 4096;
        let quantizer = NoQuantizer::<L2DistanceCalculator>::new(num_features);
        let writer = IvfWriter::new(base_directory.clone(), quantizer);

        let mut builder = IvfBuilder::new(IvfBuilderConfig {
            max_iteration: 1000,
            batch_size: 4,
            num_clusters,
            num_data_points_for_clustering: num_vectors,
            max_clusters_per_vector: 1,
            distance_threshold: 0.1,
            base_directory: base_directory.clone(),
            memory_size: 1024,
            file_size,
            num_features,
            tolerance: 0.0,
            max_posting_list_size: usize::MAX,
        })
        .expect("Failed to create builder");
        // Generate 1000 vectors of f32, dimension 4
        for i in 0..num_vectors {
            builder
                .add_vector((i + 100) as u64, &generate_random_vector(num_features))
                .expect("Vector should be added");
        }

        assert!(builder.build().is_ok());

        assert!(writer.write(&mut builder, false).is_ok());

        let quantizer = NoQuantizer::<L2DistanceCalculator>::new(num_features);
        let quantizer_directory = format!("{}/quantizer", base_directory);
        std::fs::create_dir_all(&quantizer_directory)
            .expect("Failed to create quantizer directory");
        let noq_writer = NoQuantizerWriter::new(quantizer_directory);
        assert!(noq_writer.write(&quantizer).is_ok());

        let reader = IvfReader::new(base_directory.clone());
        let index = reader
            .read::<NoQuantizer<L2DistanceCalculator>>()
            .expect("Failed to read index file");

        // Check if files were created
        assert!(fs::metadata(format!("{}/vectors", base_directory)).is_ok());
        assert!(fs::metadata(format!("{}/index", base_directory)).is_ok());

        // Verify vectors file content
        let mut context = SearchContext::new(true);
        for i in 0..num_vectors {
            let ref_vector = builder
                .vectors()
                .borrow()
                .get(i as u32)
                .expect("Failed to read vector from FileBackedAppendableVectorStorage")
                .to_vec();
            let read_vector = index
                .vector_storage
                .get(i, &mut context)
                .expect("Failed to read vector from FixedFileVectorStorage");
            assert_eq!(ref_vector.len(), read_vector.len());
            for (val_ref, val_read) in ref_vector.iter().zip(read_vector.iter()) {
                assert!((*val_ref - *val_read).abs() < f32::EPSILON);
            }
        }

        // Verify index file content
        // Verify header
        assert_eq!(index.index_storage.header().version, Version::V0);
        assert_eq!(
            index.index_storage.header().num_features,
            num_features as u32
        );
        assert_eq!(
            index.index_storage.header().num_clusters,
            num_clusters as u32
        );
        assert_eq!(index.index_storage.header().num_vectors, num_vectors as u64);
        assert_eq!(
            index.index_storage.header().centroids_len,
            (num_clusters * num_features * size_of::<f32>() + size_of::<u64>()) as u64
        );
        // Verify doc_id_mapping content
        for i in 0..num_vectors {
            let ref_id = builder.doc_id_mapping()[i];
            let read_id = index
                .index_storage
                .get_doc_id(i)
                .expect("Failed to read doc_id from FixedFileVectorStorage");
            assert_eq!(ref_id, read_id);
        }
        // Verify centroid content
        for i in 0..num_clusters {
            let ref_vector = builder
                .centroids()
                .borrow()
                .get(i as u32)
                .expect("Failed to read centroid from FileBackedAppendableVectorStorage")
                .to_vec();
            let read_vector = index
                .index_storage
                .get_centroid(i)
                .expect("Failed to read centroid from FixedFileVectorStorage");
            assert_eq!(ref_vector.len(), read_vector.len());
            for (val_ref, val_read) in ref_vector.iter().zip(read_vector.iter()) {
                assert!((*val_ref - *val_read).abs() < f32::EPSILON);
            }
        }
        // Verify posting list content
        for i in 0..num_clusters {
            let ref_vector = builder
                .posting_lists_mut()
                .get(i as u32)
                .expect("Failed to read vector from FileBackedAppendablePostingListStorage");
            let read_vector = index
                .index_storage
                .get_posting_list(i)
                .expect("Failed to read vector from FixedIndexFile");
            for (val_ref, val_read) in ref_vector.iter().zip(read_vector.iter()) {
                assert_eq!(val_ref, *val_read);
            }
        }
    }

    // Test when the max posting list size is exceeded
    #[test]
    fn test_ivf_reader_read_max_posting_list_size() {
        let temp_dir = TempDir::new("test_ivf_reader_read_max_posting_list_size")
            .expect("Failed to create temporary directory");
        let base_directory = temp_dir
            .path()
            .to_str()
            .expect("Failed to convert temporary directory path to string")
            .to_string();
        let num_clusters = 10;
        let num_vectors = 1000;
        let num_features = 4;
        let file_size = 4096;
        let quantizer = NoQuantizer::<L2DistanceCalculator>::new(num_features);
        let quantizer_directory = format!("{}/quantizer", base_directory);
        std::fs::create_dir_all(&quantizer_directory)
            .expect("Failed to create quantizer directory");
        let noq_writer = NoQuantizerWriter::new(quantizer_directory);
        assert!(noq_writer.write(&quantizer).is_ok());

        let writer = IvfWriter::new(base_directory.clone(), quantizer);

        let mut builder = IvfBuilder::new(IvfBuilderConfig {
            max_iteration: 1000,
            batch_size: 4,
            num_clusters,
            num_data_points_for_clustering: num_vectors,
            max_clusters_per_vector: 1,
            distance_threshold: 0.1,
            base_directory: base_directory.clone(),
            memory_size: 1024,
            file_size,
            num_features,
            tolerance: 0.0,
            max_posting_list_size: 10,
        })
        .expect("Failed to create builder");
        // Generate 1000 vectors of f32, dimension 4
        for i in 0..num_vectors {
            builder
                .add_vector(i as u64, &generate_random_vector(num_features))
                .expect("Vector should be added");
        }

        assert!(builder.build().is_ok());
        assert!(writer.write(&mut builder, false).is_ok());

        let reader = IvfReader::new(base_directory.clone());
        let index = reader
            .read::<NoQuantizer<L2DistanceCalculator>>()
            .expect("Failed to read index file");

        let num_centroids = index.num_clusters;

        for i in 0..num_centroids {
            // Assert that posting lists size is less than or equal to max_posting_list_size
            let posting_list = index.index_storage.get_posting_list(i);
            assert!(posting_list.is_ok());
            let posting_list = posting_list.unwrap();

            // It's possible that the posting list size is more than max_posting_list_size,
            // but it should be less than 2x.
            assert!(posting_list.len() <= 20);
        }
    }
}
