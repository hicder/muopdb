use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{Ok, Result};
use config::collection::CollectionConfig;
use config::enums::QuantizerType;
use odht::HashTableOwned;
use quantization::noq::noq::NoQuantizer;
use quantization::pq::pq::{ProductQuantizerConfig, CODEBOOK_NAME, CONFIG_FILE_NAME};
use quantization::quantization::WritableQuantizer;
use utils::bloom_filter::writer::BloomFilterWriter;
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

    fn write_common_ivf_quantizer(ivf_directory: &str, config: &CollectionConfig) -> Result<()> {
        let ivf_quantizer_directory = format!("{}/quantizer", ivf_directory);
        fs::create_dir_all(&ivf_quantizer_directory)?;
        match config.quantization_type {
            QuantizerType::ProductQuantizer => {
                // Write the config only, since it's common to all SPANNs
                // TODO(tyb) actually num_training_rows might be different across SPANNs
                let pq_config = ProductQuantizerConfig {
                    dimension: config.num_features,
                    subvector_dimension: config.product_quantization_subvector_dimension,
                    num_bits: config.product_quantization_num_bits as u8,
                };

                let config_path = Path::new(&ivf_quantizer_directory).join(CONFIG_FILE_NAME);
                if config_path.exists() {
                    // Delete the file if exists
                    std::fs::remove_file(&config_path)?;
                }

                let mut config_file = File::create(config_path)?;
                config_file.write_all(serde_yaml::to_string(&pq_config)?.as_bytes())?;
            }
            QuantizerType::NoQuantizer => {
                let ivf_quantizer = NoQuantizer::<L2DistanceCalculator>::new(config.num_features);
                ivf_quantizer.write_to_directory(&ivf_quantizer_directory)?;
            }
        };
        Ok(())
    }

    pub fn write(&self, multi_spann: &mut MultiSpannBuilder) -> Result<Vec<UserIndexInfo>> {
        let mut user_ids = multi_spann.user_ids();
        user_ids.sort();
        let base_directory = self.base_directory.clone();

        // Write bloom filter
        let bloom_filter_writer = BloomFilterWriter::new(base_directory.clone());
        if let Some(bloom_filter) = multi_spann.bloom_filter() {
            let _ = bloom_filter_writer.write(bloom_filter)?;
        }

        // Write individual SPANNs
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
                ivf_raw_vectors_offset: 0,
                ivf_raw_vectors_len: 0,
                ivf_index_offset: 0,
                ivf_index_len: 0,
                ivf_pq_codebook_offset: 0,
                ivf_pq_codebook_len: 0,
            });
        }

        // Combine them
        let centroids_directory = format!("{}/centroids", base_directory);
        fs::create_dir_all(&centroids_directory)?;
        let hnsw_directory = format!("{}/hnsw", centroids_directory);
        fs::create_dir_all(&hnsw_directory)?;

        let ivf_directory = format!("{}/ivf", base_directory);
        fs::create_dir_all(&ivf_directory)?;

        // Centroid quantizer
        let centroid_quantizer_directory = format!("{}/quantizer", centroids_directory);
        fs::create_dir_all(&centroid_quantizer_directory)?;
        let num_features = multi_spann.config().num_features;
        let no_quantizer = NoQuantizer::<L2DistanceCalculator>::new(num_features);
        no_quantizer.write_to_directory(&centroid_quantizer_directory)?;

        // IVF quantizer
        Self::write_common_ivf_quantizer(&ivf_directory, &multi_spann.config())?;

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

        // IVF raw vectors file
        let mut ivf_raw_vectors_file = File::create(format!("{}/raw_vectors", ivf_directory))?;
        let mut ivf_raw_vectors_buffer_writer = BufWriter::new(&mut ivf_raw_vectors_file);

        // IVF product quantizer codebook file
        let ivf_pq_codebook_path = format!("{}/quantizer/{}", ivf_directory, CODEBOOK_NAME);
        let mut ivf_pq_codebook_file = File::create(&ivf_pq_codebook_path)?;
        let mut ivf_pq_codebook_buffer_writer = BufWriter::new(&mut ivf_pq_codebook_file);

        let mut centroids_index_written: u64 = 0;
        let mut centroids_vector_written: u64 = 0;
        let mut ivf_index_written: u64 = 0;
        let mut ivf_vectors_written: u64 = 0;
        let mut ivf_raw_vectors_written: u64 = 0;
        let mut ivf_pq_codebook_written: u64 = 0;

        for (idx, user_id) in user_ids.iter().enumerate() {
            let user_id_base_directory = format!("{}/{}", base_directory, *user_id);
            let user_id_index_file_path =
                format!("{}/centroids/hnsw/index", user_id_base_directory);
            let user_index_info = &mut user_index_infos[idx];

            // Centroids index
            centroids_index_written += write_pad(
                centroids_index_written as usize,
                &mut hnsw_index_buffer_writer,
                16,
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
                write_pad(ivf_index_written as usize, &mut ivf_index_buffer_writer, 16)? as u64;
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
            user_index_info.ivf_raw_vectors_offset = ivf_raw_vectors_written;
            ivf_raw_vectors_written += append_file_to_writer(
                &format!("{}/ivf/raw_vectors", user_id_base_directory),
                &mut ivf_raw_vectors_buffer_writer,
            )? as u64;
            user_index_info.ivf_raw_vectors_len =
                ivf_raw_vectors_written - user_index_info.ivf_raw_vectors_offset;
            user_index_info.ivf_vectors_len =
                ivf_vectors_written - user_index_info.ivf_vectors_offset;

            // IVF product quantizer codebooks
            if multi_spann.config().quantization_type == QuantizerType::ProductQuantizer {
                ivf_pq_codebook_written += write_pad(
                    ivf_pq_codebook_written as usize,
                    &mut ivf_pq_codebook_buffer_writer,
                    8,
                )? as u64;
                user_index_info.ivf_pq_codebook_offset = ivf_pq_codebook_written;
                ivf_pq_codebook_written += append_file_to_writer(
                    &format!("{}/ivf/quantizer/codebook", user_id_base_directory),
                    &mut ivf_pq_codebook_buffer_writer,
                )? as u64;
                user_index_info.ivf_pq_codebook_len =
                    ivf_pq_codebook_written - user_index_info.ivf_pq_codebook_offset;
            }
        }

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

        // Cleanup the codebook if no quantization for IVF
        if multi_spann.config().quantization_type != QuantizerType::ProductQuantizer {
            std::fs::remove_file(ivf_pq_codebook_path).unwrap_or_default();
        }
        Ok(user_index_infos)
    }
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Seek, SeekFrom};
    use std::path::PathBuf;

    use config::collection::CollectionConfig;
    use quantization::pq::pq::CONFIG_FILE_NAME;
    use tempdir::TempDir;
    use utils::test_utils::generate_random_vector;

    use super::*;

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
        let mut collection_config = CollectionConfig::default_test_config();
        collection_config.num_features = num_features;
        collection_config.max_posting_list_size = max_posting_list_size;
        collection_config.initial_num_centroids = num_clusters;
        collection_config.posting_list_builder_vector_storage_file_size = file_size;
        collection_config.centroids_builder_vector_storage_file_size = file_size;
        collection_config.posting_list_kmeans_unbalanced_penalty = balance_factor;
        let mut builder =
            MultiSpannBuilder::new(collection_config, base_directory.clone()).unwrap();

        // Generate 1000 vectors of f32, dimension 4
        for i in 0..num_vectors {
            builder
                .insert(
                    (i % 5) as u128,
                    i as u128,
                    &generate_random_vector(num_features),
                )
                .unwrap();
        }
        builder.build().unwrap();

        let multi_spann_writer = MultiSpannWriter::new(base_directory.clone());
        let user_index_infos = multi_spann_writer.write(&mut builder).unwrap();
        assert_eq!(user_index_infos.len(), 5);

        // Check if output directories and files exist
        let bloom_filter_directory_path = format!("{}/bloom_filter", base_directory);
        let centroids_directory_path = format!("{}/centroids/hnsw", base_directory);
        let centroids_directory = PathBuf::from(&centroids_directory_path);
        let hnsw_vector_storage_path =
            format!("{}/vector_storage", centroids_directory.to_str().unwrap());
        let hnsw_index_path = format!("{}/index", centroids_directory.to_str().unwrap());

        assert!(PathBuf::from(&bloom_filter_directory_path).exists());
        assert!(centroids_directory.exists());
        assert!(PathBuf::from(&hnsw_vector_storage_path).exists());
        assert!(PathBuf::from(&hnsw_index_path).exists());
    }

    #[test]
    fn test_write_with_pq() {
        let temp_dir = TempDir::new("test_write").unwrap();

        let base_directory = temp_dir.path().to_str().unwrap().to_string();
        let num_clusters = 10;
        let num_vectors = 1000;
        let num_features = 4;
        let file_size = 4096;
        let balance_factor = 0.0;
        let max_posting_list_size = usize::MAX;
        let mut collection_config = CollectionConfig::default_test_config();
        collection_config.num_features = num_features;
        collection_config.max_posting_list_size = max_posting_list_size;
        collection_config.initial_num_centroids = num_clusters;
        collection_config.posting_list_builder_vector_storage_file_size = file_size;
        collection_config.centroids_builder_vector_storage_file_size = file_size;
        collection_config.posting_list_kmeans_unbalanced_penalty = balance_factor;
        collection_config.quantization_type = QuantizerType::ProductQuantizer;
        let mut builder =
            MultiSpannBuilder::new(collection_config, base_directory.clone()).unwrap();

        // Generate 1000 vectors of f32, dimension 4
        for i in 0..num_vectors {
            builder
                .insert(
                    (i % 5) as u128,
                    i as u128,
                    &generate_random_vector(num_features),
                )
                .unwrap();
        }
        builder.build().unwrap();

        let multi_spann_writer = MultiSpannWriter::new(base_directory.clone());
        let user_index_infos = multi_spann_writer.write(&mut builder).unwrap();
        assert_eq!(user_index_infos.len(), 5);

        // Check if output directories and files exist
        let bloom_filter_directory_path = format!("{}/bloom_filter", base_directory);
        let centroids_directory_path = format!("{}/centroids/hnsw", base_directory);
        let centroids_directory = PathBuf::from(&centroids_directory_path);
        let hnsw_vector_storage_path =
            format!("{}/vector_storage", centroids_directory.to_str().unwrap());
        let hnsw_index_path = format!("{}/index", centroids_directory.to_str().unwrap());
        let ivf_directory_path = format!("{}/ivf", base_directory);
        let ivf_quantizer_directory = format!("{}/quantizer", ivf_directory_path);
        let ivf_quantizer_codebook_path = format!("{}/codebook", ivf_quantizer_directory.clone());
        let ivf_quantizer_config_path = format!("{}/{}", ivf_quantizer_directory, CONFIG_FILE_NAME);

        assert!(PathBuf::from(&bloom_filter_directory_path).exists());
        assert!(centroids_directory.exists());
        assert!(PathBuf::from(&hnsw_vector_storage_path).exists());
        assert!(PathBuf::from(&hnsw_index_path).exists());
        assert!(PathBuf::from(&ivf_quantizer_codebook_path).exists());
        assert!(PathBuf::from(&ivf_quantizer_config_path).exists());
    }

    #[test]
    fn test_read_raw_vectors_after_writing() {
        let temp_dir = TempDir::new("test_read_raw_vectors_after_writing").unwrap();
        let base_directory = temp_dir.path().to_str().unwrap().to_string();

        let num_vectors: usize = 3;
        let num_features: usize = 4;

        // Setup collection config
        let mut collection_config = CollectionConfig::default_test_config();
        collection_config.num_features = num_features;
        collection_config.initial_num_centroids = 3;

        let mut builder =
            MultiSpannBuilder::new(collection_config, base_directory.clone()).unwrap();
        let test_vectors = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
            vec![9.0, 10.0, 11.0, 12.0],
        ];

        // Only one user
        let user_id = 42u128;
        for (i, vector) in test_vectors.iter().enumerate() {
            builder.insert(user_id, i as u128, vector).unwrap();
        }

        // Build and write the index
        builder.build().unwrap();
        let writer = MultiSpannWriter::new(base_directory.clone());
        let user_index_infos = writer.write(&mut builder).unwrap();

        // Check for one user only
        assert_eq!(user_index_infos.len(), 1);
        let user_info = &user_index_infos[0];
        assert_eq!(user_info.user_id, user_id);

        // Verify raw vectors file exists
        let raw_vectors_path = format!("{}/ivf/raw_vectors", base_directory);
        assert!(PathBuf::from(&raw_vectors_path).exists());

        // Read the raw vectors
        let file = std::fs::File::open(&raw_vectors_path).unwrap();
        let mut reader = std::io::BufReader::new(file);
        reader
            .seek(SeekFrom::Start(user_info.ivf_raw_vectors_offset))
            .unwrap();

        let mut buffer = vec![0u8; user_info.ivf_raw_vectors_len as usize];
        reader.read_exact(&mut buffer).unwrap();

        // First 8 bytes should be the number of vectors
        let num_vectors_in_file = u64::from_le_bytes([
            buffer[0], buffer[1], buffer[2], buffer[3], buffer[4], buffer[5], buffer[6], buffer[7],
        ]);
        assert_eq!(num_vectors_in_file as usize, num_vectors);

        // Read and verify each vector
        let bytes_per_float = std::mem::size_of::<f32>();
        let vector_size_bytes = num_features * bytes_per_float;

        let mut vectors = Vec::new();

        for i in 0..num_vectors {
            // 8-byte header + vector offset
            let vector_start = 8 + i * vector_size_bytes;
            let mut current_vector = Vec::new();
            for j in 0..num_features {
                let value_offset = vector_start + j * bytes_per_float;
                let value_bytes = [
                    buffer[value_offset],
                    buffer[value_offset + 1],
                    buffer[value_offset + 2],
                    buffer[value_offset + 3],
                ];
                current_vector.push(f32::from_le_bytes(value_bytes));
            }
            vectors.push(current_vector);
        }
        vectors.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap());
        assert_eq!(
            vectors, test_vectors,
            "Raw vectors do not match expected: {:?}, expected {:?}",
            vectors, test_vectors
        );
    }
}
