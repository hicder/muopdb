use std::sync::Arc;

use anyhow::Result;
use quantization::quantization::Quantizer;
use utils::file_io::env::Env;

use super::index::Spann;
use crate::hnsw::block_based::index::BlockBasedHnsw;
use crate::ivf::reader::IvfReader;

pub struct SpannReader {
    base_directory: String,
    centroids_index_offset: usize,
    centroids_vector_offset: usize,
    ivf_index_offset: usize,
    ivf_vector_offset: usize,
}

impl SpannReader {
    /// Creates a new `SpannReader` for the specified base directory.
    ///
    /// # Arguments
    /// * `base_directory` - The directory where the SPANN index files are stored.
    ///
    /// # Returns
    /// * `Self` - A new `SpannReader` instance.
    pub fn new(base_directory: String) -> Self {
        Self {
            base_directory,
            centroids_index_offset: 0,
            centroids_vector_offset: 0,
            ivf_index_offset: 0,
            ivf_vector_offset: 0,
        }
    }

    /// Creates a new `SpannReader` with specific file offsets for each component.
    ///
    /// # Arguments
    /// * `base_directory` - The directory where the SPANN index files are stored.
    /// * `centroids_index_offset` - Byte offset for the centroids HNSW index.
    /// * `centroids_vector_offset` - Byte offset for the centroids vector storage.
    /// * `ivf_index_offset` - Byte offset for the IVF index.
    /// * `ivf_vector_offset` - Byte offset for the IVF vector storage.
    ///
    /// # Returns
    /// * `Self` - A new `SpannReader` instance.
    pub fn new_with_offsets(
        base_directory: String,
        centroids_index_offset: usize,
        centroids_vector_offset: usize,
        ivf_index_offset: usize,
        ivf_vector_offset: usize,
    ) -> Self {
        Self {
            base_directory,
            centroids_index_offset,
            centroids_vector_offset,
            ivf_index_offset,
            ivf_vector_offset,
        }
    }

    /// Reads and initializes a `Spann` index from disk using Env abstraction.
    ///
    /// # Arguments
    /// * `env` - The environment for file I/O.
    ///
    /// # Returns
    /// * `Result<Spann<Q>>` - The initialized SPANN index or an error if reading fails.
    pub async fn read<Q: Quantizer>(&self, env: Arc<Box<dyn Env>>) -> Result<Spann<Q>>
    where
        Q::QuantizedT: Send + Sync,
    {
        let posting_list_path = format!("{}/ivf", self.base_directory);
        let centroid_path = format!("{}/centroids", self.base_directory);

        let centroids = BlockBasedHnsw::new_with_offsets(
            env.clone(),
            centroid_path,
            self.centroids_index_offset,
            self.centroids_vector_offset,
        )
        .await?;

        let posting_lists = IvfReader::new_with_offset(
            posting_list_path,
            self.ivf_index_offset,
            self.ivf_vector_offset,
        )
        .read::<Q>(env.clone())
        .await?;
        Ok(Spann::<_>::new(centroids, posting_lists))
    }
}

#[cfg(test)]
mod tests {
    use config::enums::QuantizerType;
    use quantization::noq::NoQuantizer;
    use quantization::pq::ProductQuantizer;
    use tempdir::TempDir;
    use utils::distance::l2::L2DistanceCalculator;
    use utils::file_io::env::{DefaultEnv, EnvConfig, FileType};
    use utils::test_utils::generate_random_vector;

    use super::*;
    use crate::spann::builder::{SpannBuilder, SpannBuilderConfig};
    use crate::spann::writer::SpannWriter;

    fn create_env() -> Arc<Box<dyn Env>> {
        let config = EnvConfig {
            file_type: FileType::CachedStandard,
            ..EnvConfig::default()
        };
        Arc::new(Box::new(DefaultEnv::new(config)))
    }

    #[tokio::test]
    async fn test_read() {
        let temp_dir = TempDir::new("test_read").unwrap();
        let base_directory = temp_dir.path().to_str().unwrap().to_string();
        let num_clusters = 5;
        let num_vectors = 100;
        let num_features = 4;
        let file_size = 4096;
        let balance_factor = 0.0;
        let max_posting_list_size = usize::MAX;
        let mut builder = SpannBuilder::new(SpannBuilderConfig {
            centroids_max_neighbors: 10,
            centroids_max_layers: 2,
            centroids_ef_construction: 100,
            centroids_vector_storage_memory_size: 1024,
            centroids_vector_storage_file_size: file_size,
            num_features,
            pq_subvector_dimension: 8,
            pq_num_bits: 8,
            pq_num_training_rows: 50,
            quantizer_type: QuantizerType::NoQuantizer,
            pq_max_iteration: 1000,
            pq_batch_size: 4,
            ivf_num_clusters: num_clusters,
            ivf_num_data_points_for_clustering: num_vectors,
            ivf_max_clusters_per_vector: 1,
            ivf_distance_threshold: 0.1,
            posting_list_encoding_type: config::enums::IntSeqEncodingType::PlainEncoding,
            ivf_base_directory: base_directory.clone(),
            ivf_vector_storage_memory_size: 1024,
            ivf_vector_storage_file_size: file_size,
            centroids_clustering_tolerance: balance_factor,
            ivf_max_posting_list_size: max_posting_list_size,
            reindex: false,
        })
        .unwrap();

        for i in 0..num_vectors {
            builder
                .add(i as u128, &generate_random_vector(num_features))
                .unwrap();
        }
        builder.build().unwrap();
        let spann_writer = SpannWriter::new(base_directory.clone());
        spann_writer.write(&mut builder).unwrap();

        let env = create_env();
        let spann_reader = SpannReader::new(base_directory.clone());
        let spann = spann_reader
            .read::<NoQuantizer<L2DistanceCalculator>>(env)
            .await
            .unwrap();

        assert_eq!(spann.count_of_all_documents(), num_vectors);
    }

    #[tokio::test]
    async fn test_read_pq() {
        let temp_dir = TempDir::new("test_read_pq").unwrap();
        let base_directory = temp_dir.path().to_str().unwrap().to_string();
        let num_clusters = 5;
        let num_vectors = 100;
        let num_features = 4;
        let file_size = 4096;
        let balance_factor = 0.0;
        let max_posting_list_size = usize::MAX;
        let mut builder = SpannBuilder::new(SpannBuilderConfig {
            centroids_max_neighbors: 10,
            centroids_max_layers: 2,
            centroids_ef_construction: 100,
            centroids_vector_storage_memory_size: 1024,
            centroids_vector_storage_file_size: file_size,
            num_features,
            pq_subvector_dimension: 2,
            pq_num_bits: 2,
            pq_num_training_rows: 50,
            quantizer_type: QuantizerType::ProductQuantizer,
            pq_max_iteration: 1000,
            pq_batch_size: 4,
            ivf_num_clusters: num_clusters,
            ivf_num_data_points_for_clustering: num_vectors,
            ivf_max_clusters_per_vector: 1,
            ivf_distance_threshold: 0.1,
            posting_list_encoding_type: config::enums::IntSeqEncodingType::PlainEncoding,
            ivf_base_directory: base_directory.clone(),
            ivf_vector_storage_memory_size: 1024,
            ivf_vector_storage_file_size: file_size,
            centroids_clustering_tolerance: balance_factor,
            ivf_max_posting_list_size: max_posting_list_size,
            reindex: false,
        })
        .unwrap();

        for i in 0..num_vectors {
            builder
                .add(i as u128, &generate_random_vector(num_features))
                .unwrap();
        }
        builder.build().unwrap();
        let spann_writer = SpannWriter::new(base_directory.clone());
        spann_writer.write(&mut builder).unwrap();

        let env = create_env();
        let spann_reader = SpannReader::new(base_directory.clone());
        let spann = spann_reader
            .read::<ProductQuantizer<L2DistanceCalculator>>(env)
            .await
            .unwrap();

        assert_eq!(spann.count_of_all_documents(), num_vectors);
    }
}
