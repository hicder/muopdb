use std::sync::Arc;

use anyhow::Result;
use compression::elias_fano::mmap_decoder::EliasFanoMmapDecoder;
use compression::noc::noc::PlainDecoder;
use config::enums::IntSeqEncodingType;
use quantization::noq::noq::NoQuantizer;
use quantization::quantization::Quantizer;
use utils::block_cache::BlockCache;
use utils::distance::l2::L2DistanceCalculator;

use super::index::Spann;
use crate::hnsw::block_based::index::BlockBasedHnsw;
use crate::hnsw::reader::HnswReader;
use crate::ivf::mmap::index::IvfType;
use crate::ivf::reader::IvfReader;

pub struct SpannReader {
    base_directory: String,
    centroids_index_offset: usize,
    centroids_vector_offset: usize,
    ivf_index_offset: usize,
    ivf_vector_offset: usize,

    ivf_type: IntSeqEncodingType,
}

impl SpannReader {
    pub fn new(base_directory: String, ivf_type: IntSeqEncodingType) -> Self {
        Self {
            base_directory,
            centroids_index_offset: 0,
            centroids_vector_offset: 0,
            ivf_index_offset: 0,
            ivf_vector_offset: 0,
            ivf_type,
        }
    }

    pub fn new_with_offsets(
        base_directory: String,
        centroids_index_offset: usize,
        centroids_vector_offset: usize,
        ivf_index_offset: usize,
        ivf_vector_offset: usize,
        ivf_type: IntSeqEncodingType,
    ) -> Self {
        Self {
            base_directory,
            centroids_index_offset,
            centroids_vector_offset,
            ivf_index_offset,
            ivf_vector_offset,
            ivf_type,
        }
    }

    pub fn read<Q: Quantizer>(&self) -> Result<Spann<Q>> {
        let posting_list_path = format!("{}/ivf", self.base_directory);
        let centroid_path = format!("{}/centroids", self.base_directory);

        let centroids = HnswReader::new_with_offset(
            centroid_path,
            self.centroids_index_offset,
            self.centroids_vector_offset,
        )
        .read::<NoQuantizer<L2DistanceCalculator>>()?;
        match self.ivf_type {
            IntSeqEncodingType::PlainEncoding => {
                let posting_lists = IvfReader::new_with_offset(
                    posting_list_path,
                    self.ivf_index_offset,
                    self.ivf_vector_offset,
                )
                .read::<Q, L2DistanceCalculator, PlainDecoder>()?;
                Ok(Spann::<_>::new(centroids, IvfType::L2Plain(posting_lists)))
            }
            IntSeqEncodingType::EliasFano => {
                let posting_lists = IvfReader::new_with_offset(
                    posting_list_path,
                    self.ivf_index_offset,
                    self.ivf_vector_offset,
                )
                .read::<Q, L2DistanceCalculator, EliasFanoMmapDecoder>()?;
                Ok(Spann::<_>::new(centroids, IvfType::L2EF(posting_lists)))
            }
        }
    }

    pub async fn read_async<Q: Quantizer>(&self, block_cache: Arc<BlockCache>) -> Result<Spann<Q>>
    where
        Q::QuantizedT: Send + Sync,
    {
        let posting_list_path = format!("{}/ivf", self.base_directory);
        let centroid_path = format!("{}/centroids", self.base_directory);

        let centroids = BlockBasedHnsw::new_with_offsets(
            block_cache.clone(),
            centroid_path,
            self.centroids_index_offset,
            self.centroids_vector_offset,
        )
        .await?;

        let posting_lists = IvfReader::new_block_based_with_offset::<Q>(
            block_cache,
            posting_list_path,
            self.ivf_index_offset,
            self.ivf_vector_offset,
        )
        .await?;
        Ok(Spann::<_>::new_async(
            centroids,
            IvfType::BlockBased(posting_lists),
        ))
    }
}

#[cfg(test)]
mod tests {

    use config::enums::{IntSeqEncodingType, QuantizerType};
    use quantization::pq::pq::ProductQuantizer;
    use tempdir::TempDir;
    use utils::mem::transmute_u8_to_slice;
    use utils::test_utils::generate_random_vector;

    use super::*;
    use crate::spann::builder::{SpannBuilder, SpannBuilderConfig};
    use crate::spann::index::Centroids;
    use crate::spann::writer::SpannWriter;

    #[test]
    fn test_read() {
        let temp_dir = TempDir::new("test_read").unwrap();
        let base_directory = temp_dir.path().to_str().unwrap().to_string();
        let num_clusters = 10;
        let num_vectors = 1000;
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
            posting_list_encoding_type: IntSeqEncodingType::PlainEncoding,
            ivf_base_directory: base_directory.clone(),
            ivf_vector_storage_memory_size: 1024,
            ivf_vector_storage_file_size: file_size,
            centroids_clustering_tolerance: balance_factor,
            ivf_max_posting_list_size: max_posting_list_size,
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

        let spann_reader =
            SpannReader::new(base_directory.clone(), IntSeqEncodingType::PlainEncoding);
        let spann = spann_reader
            .read::<NoQuantizer<L2DistanceCalculator>>()
            .unwrap();

        let centroids = spann.get_centroids();
        let posting_lists = spann.get_posting_lists();
        if let Centroids::Sync(centroids) = centroids {
            assert_eq!(
                posting_lists.num_clusters(),
                centroids.vector_storage.num_vectors()
            );
        } else {
            panic!("Expected Sync centroids");
        }
    }

    #[test]
    fn test_read_pq() {
        let temp_dir = TempDir::new("test_read_pq").unwrap();
        let base_directory = temp_dir.path().to_str().unwrap().to_string();
        let num_clusters = 10;
        let num_vectors = 1000;
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
            posting_list_encoding_type: IntSeqEncodingType::PlainEncoding,
            ivf_base_directory: base_directory.clone(),
            ivf_vector_storage_memory_size: 1024,
            ivf_vector_storage_file_size: file_size,
            centroids_clustering_tolerance: balance_factor,
            ivf_max_posting_list_size: max_posting_list_size,
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

        let spann_reader =
            SpannReader::new(base_directory.clone(), IntSeqEncodingType::PlainEncoding);
        let spann = spann_reader
            .read::<ProductQuantizer<L2DistanceCalculator>>()
            .unwrap();

        let centroids = spann.get_centroids();
        let posting_lists = spann.get_posting_lists();
        if let Centroids::Sync(centroids) = centroids {
            assert_eq!(
                posting_lists.num_clusters(),
                centroids.vector_storage.num_vectors()
            );
        } else {
            panic!("Expected Sync centroids");
        }
        // Verify posting list content
        for i in 0..num_clusters {
            let ref_vector = builder
                .ivf_builder
                .posting_lists_mut()
                .get(i as u32)
                .expect("Failed to read vector for SPANN built from builder");
            let read_vector = transmute_u8_to_slice::<u64>(
                posting_lists
                    .get_index_storage()
                    .get_posting_list(i)
                    .expect("Failed to read vector for SPANN read by reader"),
            );
            for (val_ref, val_read) in ref_vector.iter().zip(read_vector.iter()) {
                assert_eq!(val_ref, *val_read);
            }
        }
    }

    #[tokio::test]
    async fn test_read_async() {
        let temp_dir = TempDir::new("test_read_async").unwrap();
        let base_directory = temp_dir.path().to_str().unwrap().to_string();
        let num_clusters = 10;
        let num_vectors = 1000;
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
            posting_list_encoding_type: IntSeqEncodingType::EliasFano,
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
        let actual_num_clusters = builder.ivf_builder.centroids().num_vectors();
        let spann_writer = SpannWriter::new(base_directory.clone());
        spann_writer.write(&mut builder).unwrap();

        let spann_reader =
            SpannReader::new(base_directory.clone(), IntSeqEncodingType::PlainEncoding);
        let block_cache = Arc::new(BlockCache::new(
            utils::block_cache::cache::BlockCacheConfig::default(),
        ));
        let spann = spann_reader
            .read_async::<NoQuantizer<L2DistanceCalculator>>(block_cache)
            .await
            .unwrap();

        let centroids = spann.get_centroids();
        let posting_lists = spann.get_posting_lists();
        if let Centroids::Async(_) = centroids {
            assert_eq!(posting_lists.num_clusters(), actual_num_clusters);
        } else {
            panic!("Expected Async centroids");
        }
    }
}
