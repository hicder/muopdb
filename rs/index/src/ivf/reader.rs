use std::sync::Arc;

use anyhow::Result;
use compression::compression::IntSeqDecoder;
use quantization::quantization::Quantizer;
use utils::block_cache::BlockCache;
use utils::DistanceCalculator;

use crate::ivf::block_based::index::BlockBasedIvf;
use crate::ivf::mmap::index::Ivf;
use crate::posting_list::combined_file::FixedIndexFile;
use crate::posting_list::storage::PostingListStorage;
use crate::vector::fixed_file::FixedFileVectorStorage;
use crate::vector::VectorStorage;

pub struct IvfReader {
    base_directory: String,
    index_offset: usize,
    vector_offset: usize,
}

impl IvfReader {
    pub fn new(base_directory: String) -> Self {
        Self::new_with_offset(base_directory, 0, 0)
    }

    pub fn new_with_offset(
        base_directory: String,
        index_offset: usize,
        vector_offset: usize,
    ) -> Self {
        Self {
            base_directory,
            index_offset,
            vector_offset,
        }
    }

    pub fn read<Q: Quantizer, DC: DistanceCalculator, D: IntSeqDecoder>(
        &self,
    ) -> Result<Ivf<Q, DC, D>> {
        let index_storage = Box::new(PostingListStorage::FixedLocalFile(
            FixedIndexFile::new_with_offset(
                format!("{}/index", self.base_directory),
                self.index_offset,
            )?,
        ));

        let vector_storage_path = format!("{}/vectors", self.base_directory);
        let vector_storage = Box::new(VectorStorage::FixedLocalFileBacked(
            FixedFileVectorStorage::<Q::QuantizedT>::new_with_offset(
                vector_storage_path,
                index_storage.header().quantized_dimension as usize,
                self.vector_offset,
            )?,
        ));

        let num_clusters = index_storage.header().num_clusters as usize;

        // Read quantizer
        let quantizer_directory = format!("{}/quantizer", self.base_directory);
        let quantizer = Q::read(quantizer_directory).unwrap();

        Ok(Ivf::<_, DC, D>::new(
            vector_storage,
            index_storage,
            num_clusters,
            quantizer,
        ))
    }

    pub async fn new_block_based_with_offset<Q: Quantizer>(
        block_cache: Arc<BlockCache>,
        base_directory: String,
        index_offset: usize,
        vector_offset: usize,
    ) -> Result<BlockBasedIvf<Q>>
    where
        Q::QuantizedT: Send + Sync,
    {
        BlockBasedIvf::<Q>::new_with_offset(
            block_cache,
            base_directory,
            index_offset,
            vector_offset,
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::mem::size_of;

    use compression::elias_fano::ef::EliasFano;
    use compression::elias_fano::mmap_decoder::EliasFanoMmapDecoder;
    use compression::noc::noc::{PlainDecoder, PlainEncoder};
    use quantization::noq::noq::NoQuantizer;
    use quantization::pq::pq::ProductQuantizer;
    use quantization::quantization::WritableQuantizer;
    use tempdir::TempDir;
    use utils::distance::l2::L2DistanceCalculator;
    use utils::mem::transmute_u8_to_slice;
    use utils::test_utils::generate_random_vector;

    use super::*;
    use crate::ivf::builder::{IvfBuilder, IvfBuilderConfig};
    use crate::ivf::writer::IvfWriter;
    use crate::posting_list::combined_file::Version;

    /// Tests the `IvfReader`'s ability to correctly read an IVF index that uses Elias-Fano encoding for posting lists.
    ///
    /// This test performs the following steps:
    /// 1. Creates a temporary directory for storing the index files.
    /// 2. Initializes an `IvfWriter` with `EliasFano` encoding.
    /// 3. Creates an `IvfBuilder` to build the IVF index with randomly generated vectors.
    /// 4. Writes the built index to disk using the `IvfWriter`.
    /// 5. Initializes an `IvfReader` to read the index from disk.
    /// 6. Reads the index using the `IvfReader`.
    /// 7. Verifies the integrity of the read index by checking:
    ///     - The existence of the vector and index files.
    ///     - The consistency of vectors between the builder and the read index.
    ///     - The correctness of index file headers.
    ///     - The correctness of document ID mappings.
    ///     - The consistency of centroid content.
    ///     - The consistency of posting list content, ensuring that the Elias-Fano encoded
    ///       posting lists are correctly decoded and match the original data.
    #[test]
    fn test_ivf_reader_elias_fano() {
        let temp_dir = TempDir::new("test_ivf_reader_elias_fano")
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
        assert!(quantizer.write_to_directory(&quantizer_directory).is_ok());
        let writer =
            IvfWriter::<_, EliasFano, L2DistanceCalculator>::new(base_directory.clone(), quantizer);

        let mut builder: IvfBuilder<L2DistanceCalculator> = IvfBuilder::new(IvfBuilderConfig {
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
                .add_vector((i + 100) as u128, &generate_random_vector(num_features))
                .expect("Vector should be added");
        }

        assert!(builder.build().is_ok());

        assert!(writer.write(&mut builder, false).is_ok());

        let reader = IvfReader::new(base_directory.clone());
        let index = reader
            .read::<NoQuantizer<L2DistanceCalculator>, L2DistanceCalculator, EliasFanoMmapDecoder>()
            .expect("Failed to read index file");

        // Check if files were created
        assert!(fs::metadata(format!("{}/vectors", base_directory)).is_ok());
        assert!(fs::metadata(format!("{}/index", base_directory)).is_ok());

        // Verify vectors file content
        for i in 0..num_vectors {
            let ref_vector = builder
                .vectors()
                .get_no_context(i as u32)
                .expect("Failed to read vector from FileBackedAppendableVectorStorage")
                .to_vec();
            let read_vector = index
                .vector_storage
                .get_no_context(i as u32)
                .expect("Failed to read vector from FixedFileVectorStorage");
            assert_eq!(ref_vector.len(), read_vector.len());
            for (val_ref, val_read) in ref_vector.iter().zip(read_vector.iter()) {
                assert!((*val_ref - *val_read).abs() < f32::EPSILON);
            }
        }

        // Verify index file content
        // Verify header
        assert_eq!(index.posting_list_storage.header().version, Version::V0);
        assert_eq!(
            index.posting_list_storage.header().num_features,
            num_features as u32
        );
        assert_eq!(
            index.posting_list_storage.header().num_clusters,
            num_clusters as u32
        );
        assert_eq!(
            index.posting_list_storage.header().num_vectors,
            num_vectors as u64
        );
        assert_eq!(
            index.posting_list_storage.header().centroids_len,
            (num_clusters * num_features * size_of::<f32>() + size_of::<u64>()) as u64
        );
        // Verify doc_id_mapping content
        for i in 0..num_vectors {
            let ref_id = builder.doc_id_mapping()[i];
            let read_id = index
                .posting_list_storage
                .get_doc_id(i)
                .expect("Failed to read doc_id from FixedFileVectorStorage");
            assert_eq!(ref_id, read_id);
        }
        // Verify centroid content
        for i in 0..num_clusters {
            let ref_vector = builder
                .centroids()
                .get_no_context(i as u32)
                .expect("Failed to read centroid from FileBackedAppendableVectorStorage")
                .to_vec();
            let read_vector = index
                .posting_list_storage
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
            let byte_slice = index
                .posting_list_storage
                .get_posting_list(i)
                .expect("Failed to read vector from FixedIndexFile");
            let decoder = EliasFanoMmapDecoder::<u64>::new_decoder(byte_slice)
                .expect("Failed to create posting list decoder");
            for (val_ref, val_read) in ref_vector.iter().zip(decoder.get_iterator(byte_slice)) {
                assert_eq!(val_ref, val_read);
            }
        }
    }

    /// Tests the `IvfReader`'s ability to read an IVF index that was encoded using Elias-Fano,
    /// and compares its search results with a reference index encoded using Plain encoding.
    ///
    /// The purpose of this test is to ensure that the Elias-Fano encoding and decoding process
    /// does not introduce any discrepancies in the search results compared to a baseline.
    ///
    /// This test performs the following steps:
    /// 1. Creates two temporary directories: one for the reference index (Plain encoding) and
    ///    another for the index to be tested (Elias-Fano encoding).
    /// 2. Initializes two `IvfWriter` instances, one with `PlainEncoder` and another with
    ///    `EliasFano`.
    /// 3. Creates an `IvfBuilder` and adds randomly generated vectors to it.
    /// 4. Builds the IVF index using the builder.
    /// 5. Writes both the reference index and the Elias-Fano encoded index to their
    ///    respective directories.
    /// 6. Initializes two `IvfReader` instances, one for the reference index and another for
    ///    the Elias-Fano encoded index.
    /// 7. Reads both indices using their respective readers.
    /// 8. Generates random queries and performs searches on both indices.
    /// 9. Compares the search results (IDs and scores) of both indices to ensure they are identical.
    #[tokio::test]
    async fn test_ivf_reader_read_elias_fano_encoding() {
        // Create reference index (using PlainEncoder/Decoder)
        let temp_dir_ref = TempDir::new("test_ivf_reader_read_elias_fano_encoding_ref")
            .expect("Failed to create ref temporary directory");
        let base_directory_ref = temp_dir_ref
            .path()
            .to_str()
            .expect("Failed to convert ref temporary directory path to string")
            .to_string();
        // Create index using EliasFano
        let temp_dir = TempDir::new("test_ivf_reader_read_elias_fano_encoding")
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
        let quantizer_directory_ref = format!("{}/quantizer", base_directory_ref);
        std::fs::create_dir_all(&quantizer_directory_ref)
            .expect("Failed to create quantizer directory");
        assert!(quantizer
            .write_to_directory(&quantizer_directory_ref)
            .is_ok());
        let writer_ref = IvfWriter::<_, PlainEncoder, L2DistanceCalculator>::new(
            base_directory_ref.clone(),
            quantizer,
        );
        let quantizer = NoQuantizer::<L2DistanceCalculator>::new(num_features);
        let quantizer_directory = format!("{}/quantizer", base_directory);
        std::fs::create_dir_all(&quantizer_directory)
            .expect("Failed to create quantizer directory");
        assert!(quantizer.write_to_directory(&quantizer_directory).is_ok());
        let writer =
            IvfWriter::<_, EliasFano, L2DistanceCalculator>::new(base_directory.clone(), quantizer);

        let mut builder: IvfBuilder<L2DistanceCalculator> = IvfBuilder::new(IvfBuilderConfig {
            max_iteration: 1000,
            batch_size: 4,
            num_clusters,
            num_data_points_for_clustering: num_vectors,
            max_clusters_per_vector: 1,
            distance_threshold: 0.1,
            base_directory: base_directory_ref.clone(),
            memory_size: 1024,
            file_size,
            num_features,
            tolerance: 0.0,
            max_posting_list_size: usize::MAX,
        })
        .expect("Failed to create builder");

        // Generate 1000 vectors of f32, dimension 4
        for i in 0..num_vectors {
            let vector = generate_random_vector(num_features);
            builder
                .add_vector((i + 100) as u128, &vector)
                .expect("Vector should be added");
        }

        assert!(builder.build().is_ok());

        assert!(writer_ref.write(&mut builder, false).is_ok());
        assert!(writer.write(&mut builder, false).is_ok());

        let reader_ref = IvfReader::new(base_directory_ref.clone());
        let index_ref = reader_ref
            .read::<NoQuantizer<L2DistanceCalculator>, L2DistanceCalculator, PlainDecoder>()
            .expect("Failed to read ref index file");
        let reader = IvfReader::new(base_directory.clone());
        let index = reader
            .read::<NoQuantizer<L2DistanceCalculator>, L2DistanceCalculator, EliasFanoMmapDecoder>()
            .expect("Failed to read index file");

        let k = 3;
        let num_probes = 2;
        // Generate 1000 queries
        for _ in 0..1000 {
            let query = generate_random_vector(num_features);
            let results_ref = index_ref
                .search(&query, k, num_probes, false, None)
                .await
                .expect("IVF search ref should return a result");
            let results = index
                .search(&query, k, num_probes, false, None)
                .await
                .expect("IVF search should return a result");
            assert_eq!(results_ref.id_with_scores, results.id_with_scores);
        }
    }

    /// Tests the `IvfReader`'s ability to read a basic IVF index.
    ///
    /// This test performs the following steps:
    /// 1.  Sets up a temporary directory and configurations for the IVF index.
    /// 2.  Initializes a `NoQuantizer`.
    /// 3.  Creates an `IvfWriter` to write the index to disk.
    /// 4.  Builds an IVF index using `IvfBuilder` with randomly generated vectors.
    /// 5.  Writes the built index to disk using the `IvfWriter`.
    /// 6.  Creates an `IvfReader` to read the index from disk.
    /// 7.  Reads the index using the `IvfReader`.
    /// 8.  Verifies the integrity of the read index by checking:
    ///     - The existence of the vector and index files.
    ///     - The consistency of vectors.
    ///     - The correctness of index file headers.
    ///     - The correctness of document ID mappings.
    ///     - The consistency of centroid content.
    ///     - The consistency of posting list content.
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
        let quantizer_directory = format!("{}/quantizer", base_directory);
        std::fs::create_dir_all(&quantizer_directory)
            .expect("Failed to create quantizer directory");
        assert!(quantizer.write_to_directory(&quantizer_directory).is_ok());
        let writer = IvfWriter::<_, PlainEncoder, L2DistanceCalculator>::new(
            base_directory.clone(),
            quantizer,
        );

        let mut builder: IvfBuilder<L2DistanceCalculator> = IvfBuilder::new(IvfBuilderConfig {
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
                .add_vector((i + 100) as u128, &generate_random_vector(num_features))
                .expect("Vector should be added");
        }

        assert!(builder.build().is_ok());

        assert!(writer.write(&mut builder, false).is_ok());

        let quantizer = NoQuantizer::<L2DistanceCalculator>::new(num_features);
        let quantizer_directory = format!("{}/quantizer", base_directory);
        std::fs::create_dir_all(&quantizer_directory)
            .expect("Failed to create quantizer directory");
        assert!(quantizer.write_to_directory(&quantizer_directory).is_ok());

        let reader = IvfReader::new(base_directory.clone());
        let index = reader
            .read::<NoQuantizer<L2DistanceCalculator>, L2DistanceCalculator, PlainDecoder>()
            .expect("Failed to read index file");

        // Check if files were created
        assert!(fs::metadata(format!("{}/vectors", base_directory)).is_ok());
        assert!(fs::metadata(format!("{}/index", base_directory)).is_ok());

        // Verify vectors file content
        for i in 0..num_vectors {
            let ref_vector = builder
                .vectors()
                .get_no_context(i as u32)
                .expect("Failed to read vector from FileBackedAppendableVectorStorage")
                .to_vec();
            let read_vector = index
                .vector_storage
                .get_no_context(i as u32)
                .expect("Failed to read vector from FixedFileVectorStorage");
            assert_eq!(ref_vector.len(), read_vector.len());
            for (val_ref, val_read) in ref_vector.iter().zip(read_vector.iter()) {
                assert!((*val_ref - *val_read).abs() < f32::EPSILON);
            }
        }

        // Verify index file content
        // Verify header
        assert_eq!(index.posting_list_storage.header().version, Version::V0);
        assert_eq!(
            index.posting_list_storage.header().num_features,
            num_features as u32
        );
        assert_eq!(
            index.posting_list_storage.header().num_clusters,
            num_clusters as u32
        );
        assert_eq!(
            index.posting_list_storage.header().num_vectors,
            num_vectors as u64
        );
        assert_eq!(
            index.posting_list_storage.header().centroids_len,
            (num_clusters * num_features * size_of::<f32>() + size_of::<u64>()) as u64
        );
        // Verify doc_id_mapping content
        for i in 0..num_vectors {
            let ref_id = builder.doc_id_mapping()[i];
            let read_id = index
                .posting_list_storage
                .get_doc_id(i)
                .expect("Failed to read doc_id from FixedFileVectorStorage");
            assert_eq!(ref_id, read_id);
        }
        // Verify centroid content
        for i in 0..num_clusters {
            let ref_vector = builder
                .centroids()
                .get_no_context(i as u32)
                .expect("Failed to read centroid from FileBackedAppendableVectorStorage")
                .to_vec();
            let read_vector = index
                .posting_list_storage
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
            let read_vector = transmute_u8_to_slice::<u64>(
                index
                    .posting_list_storage
                    .get_posting_list(i)
                    .expect("Failed to read vector from FixedIndexFile"),
            );
            for (val_ref, val_read) in ref_vector.iter().zip(read_vector.iter()) {
                assert_eq!(val_ref, *val_read);
            }
        }
    }

    /// Tests the IvfReader's ability to read an index that uses Product Quantization (PQ).
    ///
    /// This test verifies that:
    /// 1. The index is properly written with ProductQuantizer
    /// 2. Files are created correctly (vectors and index files)
    /// 3. Vector storage content is consistent after quantization
    /// 4. Index file components are correct:
    ///    - Header information (version, dimensions, clusters, etc.)
    ///    - Document ID mapping
    ///    - Centroid content
    ///    - Posting list content
    ///
    /// The test uses a random codebook for quantization, ensuring the same codebook
    /// is used for both writing and reading to maintain consistency. The vector
    /// comparisons account for quantization effects by comparing quantized versions
    /// rather than original vectors.
    #[test]
    fn test_ivf_reader_read_product_quantizer() {
        let temp_dir = TempDir::new("test_ivf_reader_read_product_quantizer")
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
        let num_subvectors = 2;
        let num_clusters_per_subvector = 256;

        // Initialize codebook for product quantizer with random values
        let codebook_size =
            num_subvectors * num_clusters_per_subvector * (num_features / num_subvectors);
        let codebook: Vec<f32> = (0..codebook_size)
            .map(|_| generate_random_vector(1)[0])
            .collect();
        let quantizer = ProductQuantizer::<L2DistanceCalculator>::new(
            num_features,
            num_features / num_subvectors, // subvector_dimension
            8,                             // num_bits (8 bits = 256 clusters per subvector)
            codebook.clone(),              // Clone because we need it for the second initialization
            base_directory.clone(),
        )
        .expect("Failed to create product quantizer");
        let quantizer_directory = format!("{}/quantizer", base_directory);
        std::fs::create_dir_all(&quantizer_directory)
            .expect("Failed to create quantizer directory");
        assert!(quantizer.write_to_directory(&quantizer_directory).is_ok());
        let writer = IvfWriter::<_, PlainEncoder, L2DistanceCalculator>::new(
            base_directory.clone(),
            quantizer,
        );

        let mut builder: IvfBuilder<L2DistanceCalculator> = IvfBuilder::new(IvfBuilderConfig {
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
                .add_vector((i + 100) as u128, &generate_random_vector(num_features))
                .expect("Vector should be added");
        }

        assert!(builder.build().is_ok());
        assert!(writer.write(&mut builder, false).is_ok());

        // Use same codebook for consistency
        let quantizer = ProductQuantizer::<L2DistanceCalculator>::new(
            num_features,
            num_features / num_subvectors, // subvector_dimension
            8,                             // num_bits (8 bits = 256 clusters per subvector)
            codebook,                      // Use the same codebook as before
            base_directory.clone(),
        )
        .expect("Failed to create product quantizer");
        let quantizer_directory = format!("{}/quantizer", base_directory);
        std::fs::create_dir_all(&quantizer_directory)
            .expect("Failed to create quantizer directory");
        assert!(quantizer.write_to_directory(&quantizer_directory).is_ok());

        let reader = IvfReader::new(base_directory.clone());
        let index = reader
            .read::<ProductQuantizer<L2DistanceCalculator>, L2DistanceCalculator, PlainDecoder>()
            .expect("Failed to read index file");

        // Check if files were created
        assert!(fs::metadata(format!("{}/vectors", base_directory)).is_ok());
        assert!(fs::metadata(format!("{}/index", base_directory)).is_ok());

        // Verify vectors file content
        for i in 0..num_vectors {
            let ref_vector = builder
                .vectors()
                .get_no_context(i as u32)
                .expect("Failed to read vector from FileBackedAppendableVectorStorage")
                .to_vec();
            let read_vector = index
                .vector_storage
                .get_no_context(i as u32)
                .expect("Failed to read vector from FixedFileVectorStorage");
            // For product quantizer, we need to compare the quantized vectors
            let quantized_ref = index.quantizer.quantize(&ref_vector);
            assert_eq!(quantized_ref.len(), read_vector.len());
            for (val_ref, val_read) in quantized_ref.iter().zip(read_vector.iter()) {
                assert_eq!(val_ref, val_read);
            }
        }

        // Verify index file content
        // Verify header
        assert_eq!(index.posting_list_storage.header().version, Version::V0);
        assert_eq!(
            index.posting_list_storage.header().num_features,
            num_features as u32
        );
        assert_eq!(
            index.posting_list_storage.header().num_clusters,
            num_clusters as u32
        );
        assert_eq!(
            index.posting_list_storage.header().num_vectors,
            num_vectors as u64
        );
        assert_eq!(
            index.posting_list_storage.header().centroids_len,
            (num_clusters * num_features * size_of::<f32>() + size_of::<u64>()) as u64
        );

        // Verify doc_id_mapping content
        for i in 0..num_vectors {
            let ref_id = builder.doc_id_mapping()[i];
            let read_id = index
                .posting_list_storage
                .get_doc_id(i)
                .expect("Failed to read doc_id from FixedFileVectorStorage");
            assert_eq!(ref_id, read_id);
        }

        // Verify centroid content
        for i in 0..num_clusters {
            let ref_vector = builder
                .centroids()
                .get_no_context(i as u32)
                .expect("Failed to read centroid from FileBackedAppendableVectorStorage")
                .to_vec();
            let read_vector = index
                .posting_list_storage
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
            let read_vector = transmute_u8_to_slice::<u64>(
                index
                    .posting_list_storage
                    .get_posting_list(i)
                    .expect("Failed to read vector from FixedIndexFile"),
            );
            for (val_ref, val_read) in ref_vector.iter().zip(read_vector.iter()) {
                assert_eq!(val_ref, *val_read);
            }
        }
    }

    /// Tests the IvfReader`s ability to limit the size of posting lists during index creation.
    ///
    /// This test verifies that the `max_posting_list_size` configuration parameter in `IvfBuilderConfig`
    /// correctly restricts the number of vectors added to each posting list.
    /// It checks if the posting list sizes in the created index are within the expected bounds, ensuring
    /// that they do not exceed `max_posting_list_size` by a significant factor.
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
        assert!(quantizer.write_to_directory(&quantizer_directory).is_ok());

        let writer = IvfWriter::<_, PlainEncoder, L2DistanceCalculator>::new(
            base_directory.clone(),
            quantizer,
        );

        let mut builder: IvfBuilder<L2DistanceCalculator> = IvfBuilder::new(IvfBuilderConfig {
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
                .add_vector(i as u128, &generate_random_vector(num_features))
                .expect("Vector should be added");
        }

        assert!(builder.build().is_ok());
        assert!(writer.write(&mut builder, false).is_ok());

        let reader = IvfReader::new(base_directory.clone());
        let index = reader
            .read::<NoQuantizer<L2DistanceCalculator>, L2DistanceCalculator, PlainDecoder>()
            .expect("Failed to read index file");

        let num_centroids = index.num_clusters;

        for i in 0..num_centroids {
            // Assert that posting lists size is less than or equal to max_posting_list_size
            let posting_list_byte_arr = index.posting_list_storage.get_posting_list(i);
            assert!(posting_list_byte_arr.is_ok());
            let posting_list = transmute_u8_to_slice::<u64>(posting_list_byte_arr.unwrap());

            // It's possible that the posting list size is more than max_posting_list_size,
            // but it should be less than 3x.
            assert!(posting_list.len() <= 30);
        }
    }
}
