use std::fs::{self, File};
use std::io::BufWriter;

use anyhow::{Ok, Result};
use odht::HashTableOwned;
use quantization::noq::noq::{NoQuantizer, NoQuantizerWriter};
use utils::distance::l2::L2DistanceCalculator;
use utils::io::{append_file_to_writer, write_pad};

use super::user_index_info::{HashConfig, UserIndexInfo};
use crate::multi_spann::builder::MultiSpannBuilder;
use crate::spann::writer::SpannWriter;

pub struct MultiSpannWriter {
    base_directory: String,
}

impl MultiSpannWriter {
    pub fn new(base_directory: String) -> Self {
        Self { base_directory }
    }

    pub fn write(&self, multi_spann: &mut MultiSpannBuilder) -> Result<Vec<UserIndexInfo>> {
        let mut user_ids = multi_spann.user_ids();
        user_ids.sort();
        let base_directory = self.base_directory.clone();

        // Build individual spanns
        for user_id in user_ids.iter() {
            let spann_directory = format!("{}/{}", base_directory, *user_id);
            let spann_writer = SpannWriter::new(spann_directory);
            let mut spann_builder = multi_spann.take_builder_for_user(*user_id).unwrap();
            spann_writer.write(&mut spann_builder)?;
        }

        let mut user_index_infos = Vec::with_capacity(user_ids.len());
        for user_id in user_ids.iter() {
            user_index_infos.push(UserIndexInfo {
                user_id: *user_id,
                centroid_vector_offset: 0,
                centroid_vector_len: 0,
                centroid_index_offset: 0,
                centroid_index_len: 0,
                ivf_vectors_offset: 0,
                ivf_vectors_len: 0,
                ivf_index_offset: 0,
                ivf_index_len: 0,
            });
        }

        // Combine them
        let centroids_directory = format!("{}/centroids", base_directory);
        fs::create_dir_all(&centroids_directory)?;
        let hnsw_directory = format!("{}/hnsw", centroids_directory);
        fs::create_dir_all(&hnsw_directory)?;

        let ivf_directory = format!("{}/ivf", base_directory);
        fs::create_dir_all(&ivf_directory)?;

        // Centroids index
        let mut hnsw_index_file = File::create(format!("{}/index", hnsw_directory))?;
        let mut hnsw_index_buffer_writer = BufWriter::new(&mut hnsw_index_file);

        // Centroids vector file
        let mut centroids_vector_file = File::create(format!("{}/vector_storage", hnsw_directory))?;
        let mut centroids_vector_buffer_writer = BufWriter::new(&mut centroids_vector_file);

        // IVF index file
        let mut ivf_index_file = File::create(format!("{}/index", ivf_directory))?;
        let mut ivf_index_buffer_writer = BufWriter::new(&mut ivf_index_file);

        // IVF vectors file
        let mut ivf_vectors_file = File::create(format!("{}/vectors", ivf_directory))?;
        let mut ivf_vectors_buffer_writer = BufWriter::new(&mut ivf_vectors_file);

        let mut centroids_index_written: u64 = 0;
        let mut centroids_vector_written: u64 = 0;
        let mut ivf_index_written: u64 = 0;
        let mut ivf_vectors_written: u64 = 0;

        for (idx, user_id) in user_ids.iter().enumerate() {
            let user_id_base_directory = format!("{}/{}", base_directory, *user_id);
            let user_id_index_file_path =
                format!("{}/centroids/hnsw/index", user_id_base_directory);
            let user_index_info = &mut user_index_infos[idx];

            // Centroids index
            centroids_index_written += write_pad(
                centroids_index_written as usize,
                &mut hnsw_index_buffer_writer,
                8,
            )? as u64;
            user_index_info.centroid_index_offset = centroids_index_written;
            centroids_index_written +=
                append_file_to_writer(&user_id_index_file_path, &mut hnsw_index_buffer_writer)?
                    as u64;
            user_index_info.centroid_index_len =
                centroids_index_written - user_index_info.centroid_index_offset;

            // Centroids vector
            centroids_vector_written += write_pad(
                centroids_vector_written as usize,
                &mut centroids_vector_buffer_writer,
                8,
            )? as u64;
            user_index_info.centroid_vector_offset = centroids_vector_written;
            centroids_vector_written += append_file_to_writer(
                &format!("{}/centroids/hnsw/vector_storage", user_id_base_directory),
                &mut centroids_vector_buffer_writer,
            )? as u64;
            user_index_info.centroid_vector_len =
                centroids_vector_written - user_index_info.centroid_vector_offset;

            // IVF index
            ivf_index_written +=
                write_pad(ivf_index_written as usize, &mut ivf_index_buffer_writer, 8)? as u64;
            user_index_info.ivf_index_offset = ivf_index_written;
            ivf_index_written += append_file_to_writer(
                &format!("{}/ivf/index", user_id_base_directory),
                &mut ivf_index_buffer_writer,
            )? as u64;
            user_index_info.ivf_index_len = ivf_index_written - user_index_info.ivf_index_offset;

            // IVF vectors
            ivf_vectors_written += write_pad(
                ivf_vectors_written as usize,
                &mut ivf_vectors_buffer_writer,
                8,
            )? as u64;
            user_index_info.ivf_vectors_offset = ivf_vectors_written;
            ivf_vectors_written += append_file_to_writer(
                &format!("{}/ivf/vectors", user_id_base_directory),
                &mut ivf_vectors_buffer_writer,
            )? as u64;
            user_index_info.ivf_vectors_len =
                ivf_vectors_written - user_index_info.ivf_vectors_offset;
        }

        // Centroid quantizer
        let centroid_quantizer_directory = format!("{}/quantizer", centroids_directory);
        fs::create_dir_all(&centroid_quantizer_directory)?;
        let num_features = multi_spann.config().num_features;
        let no_quantizer = NoQuantizer::<L2DistanceCalculator>::new(num_features);
        let quantizer_writer = NoQuantizerWriter::new(centroid_quantizer_directory);
        quantizer_writer.write(&no_quantizer)?;

        // IVF quantizer
        let ivf_quantizer_directory = format!("{}/quantizer", ivf_directory);
        fs::create_dir_all(&ivf_quantizer_directory)?;
        let ivf_quantizer = NoQuantizer::<L2DistanceCalculator>::new(num_features);
        let ivf_quantizer_writer = NoQuantizerWriter::new(ivf_quantizer_directory);
        ivf_quantizer_writer.write(&ivf_quantizer)?;

        // Write user index infos
        let mut hash_table = HashTableOwned::<HashConfig>::with_capacity(user_ids.len(), 90);
        for user_index_info in user_index_infos.iter() {
            hash_table.insert(&user_index_info.user_id, &user_index_info);
        }
        let serialized = hash_table.raw_bytes();
        let user_index_info_file = format!("{}/user_index_info", base_directory);
        fs::write(&user_index_info_file, serialized)?;

        // Cleanup the user directories
        for user_id in user_ids.iter() {
            let user_id_base_directory = format!("{}/{}", base_directory, *user_id);

            // It's ok to fail for some reason
            std::fs::remove_dir_all(user_id_base_directory).unwrap_or_default();
        }

        Ok(user_index_infos)
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
        let mut builder = MultiSpannBuilder::new(SpannBuilderConfig {
            max_neighbors: 10,
            max_layers: 2,
            ef_construction: 100,
            vector_storage_memory_size: 1024,
            vector_storage_file_size: file_size,
            num_features,
            subvector_dimension: 8,
            num_bits: 8,
            max_iteration: 1000,
            batch_size: 4,
            quantizer_type: QuantizerType::NoQuantizer,
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
                .insert(
                    (i % 5) as u64,
                    i as u64,
                    &generate_random_vector(num_features),
                )
                .unwrap();
        }
        builder.build().unwrap();

        let multi_spann_writer = MultiSpannWriter::new(base_directory.clone());
        let user_index_infos = multi_spann_writer.write(&mut builder).unwrap();
        assert_eq!(user_index_infos.len(), 5);

        // Check if output directories and files exist
        let centroids_directory_path = format!("{}/centroids/hnsw", base_directory);
        let centroids_directory = PathBuf::from(&centroids_directory_path);
        let hnsw_vector_storage_path =
            format!("{}/vector_storage", centroids_directory.to_str().unwrap());
        let hnsw_index_path = format!("{}/index", centroids_directory.to_str().unwrap());

        assert!(PathBuf::from(&centroids_directory_path).exists());
        assert!(PathBuf::from(&hnsw_vector_storage_path).exists());
        assert!(PathBuf::from(&hnsw_index_path).exists());
    }
}
