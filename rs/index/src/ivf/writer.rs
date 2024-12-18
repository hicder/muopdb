use std::fs::{create_dir_all, remove_file, File};
use std::io::{BufWriter, Write};

use anyhow::{anyhow, Context, Result};
use log::debug;
use quantization::quantization::Quantizer;
use quantization::typing::VectorOps;
use utils::io::{append_file_to_writer, wrap_write};

use crate::ivf::builder::IvfBuilder;
use crate::posting_list::combined_file::{Header, Version};
use crate::vector::file::FileBackedAppendableVectorStorage;
use crate::vector::VectorStorage;

pub struct IvfWriter<Q: Quantizer> {
    base_directory: String,
    quantizer: Q,
}

impl<Q: Quantizer> IvfWriter<Q> {
    pub fn new(base_directory: String, quantizer: Q) -> Self {
        Self {
            base_directory,
            quantizer,
        }
    }

    pub fn write(&self, ivf_builder: &mut IvfBuilder, reindex: bool) -> Result<()> {
        if reindex {
            // Reindex the vectors for efficient lookup
            ivf_builder
                .reindex()
                .context("failed to reindex during write")?;
            debug!("Finish reindexing");
        }

        let num_features = ivf_builder.config().num_features;
        let num_clusters = ivf_builder.centroids().len();
        let num_vectors = ivf_builder.vectors().len();

        // Write vectors
        let vectors_len = self
            .quantize_and_write_vectors(ivf_builder)
            .context("Failed to write vectors")?;
        let expected_bytes_written = std::mem::size_of::<u64>()
            + std::mem::size_of::<Q::QuantizedT>() * num_features * num_vectors;
        if vectors_len != expected_bytes_written {
            return Err(anyhow!(
                "Expected to write {} bytes in vector storage, but wrote {}",
                expected_bytes_written,
                vectors_len,
            ));
        }

        // Write doc_id_mapping
        let doc_id_mapping_len = self
            .write_doc_id_mapping(ivf_builder)
            .context("Failed to write doc_id_mapping")?;
        let expected_bytes_written = std::mem::size_of::<u64>() * (num_vectors + 1);
        if doc_id_mapping_len != expected_bytes_written {
            return Err(anyhow!(
                "Expected to write {} bytes in centroid storage, but wrote {}",
                expected_bytes_written,
                doc_id_mapping_len,
            ));
        }

        // Write centroids
        let centroids_len = self
            .write_centroids(ivf_builder)
            .context("Failed to write centroids")?;
        let expected_bytes_written =
            std::mem::size_of::<u64>() + std::mem::size_of::<f32>() * num_features * num_clusters;
        if centroids_len != expected_bytes_written {
            return Err(anyhow!(
                "Expected to write {} bytes in centroid storage, but wrote {}",
                expected_bytes_written,
                centroids_len,
            ));
        }

        // Write posting_lists
        let posting_lists_len = self
            .write_posting_lists(ivf_builder)
            .context("Failed to write posting_lists")?;
        let expected_bytes_written = std::mem::size_of::<u64>() * 2 * num_clusters
            + std::mem::size_of::<u64>() * num_vectors
            + std::mem::size_of::<u64>();
        if posting_lists_len != expected_bytes_written {
            return Err(anyhow!(
                "Expected to write {} bytes in posting list storage, but wrote {}",
                expected_bytes_written,
                posting_lists_len,
            ));
        }

        let header: Header = Header {
            version: Version::V0,
            num_features: num_features as u32,
            num_clusters: num_clusters as u32,
            num_vectors: num_vectors as u64,
            doc_id_mapping_len: doc_id_mapping_len as u64,
            centroids_len: centroids_len as u64,
            posting_lists_len: posting_lists_len as u64,
        };

        self.combine_files(&header)?;
        Ok(())
    }

    fn quantize_and_write_vectors(&self, ivf_builder: &IvfBuilder) -> Result<usize> {
        // Quantize vectors
        let full_vectors = &ivf_builder.vectors();
        let config = ivf_builder.config();
        let quantized_vectors_path = format!("{}/quantized/vector_storage", config.base_directory);
        create_dir_all(&quantized_vectors_path)?;
        let mut quantized_vectors =
            Box::new(FileBackedAppendableVectorStorage::<Q::QuantizedT>::new(
                quantized_vectors_path,
                config.memory_size,
                config.file_size,
                config.num_features,
            ));

        for i in 0..full_vectors.len() {
            let vector = full_vectors.get(i as u32)?;
            let quantized_vector = Q::QuantizedT::process_vector(vector, &self.quantizer);
            quantized_vectors.append(&quantized_vector)?;
        }

        let path = format!("{}/vectors", self.base_directory);
        let mut file = File::create(path)?;
        let mut writer = BufWriter::new(&mut file);

        let bytes_written = quantized_vectors.write(&mut writer)?;
        Ok(bytes_written)
    }

    fn write_doc_id_mapping(&self, ivf_builder: &IvfBuilder) -> Result<usize> {
        let path = format!("{}/doc_id_mapping", self.base_directory);
        let mut file = File::create(path)?;
        let mut writer = BufWriter::new(&mut file);

        let mut bytes_written = wrap_write(
            &mut writer,
            &(ivf_builder.doc_id_mapping().len() as u64).to_le_bytes(),
        )?;
        for doc_id in ivf_builder.doc_id_mapping() {
            bytes_written += wrap_write(&mut writer, &doc_id.to_le_bytes())?;
        }
        Ok(bytes_written)
    }

    fn write_centroids(&self, ivf_builder: &IvfBuilder) -> Result<usize> {
        let path = format!("{}/centroids", self.base_directory);
        let mut file = File::create(path)?;
        let mut writer = BufWriter::new(&mut file);

        let bytes_written = ivf_builder.centroids().write(&mut writer)?;
        Ok(bytes_written)
    }

    fn write_posting_lists(&self, ivf_builder: &mut IvfBuilder) -> Result<usize> {
        let path = format!("{}/posting_lists", self.base_directory);
        let mut file = File::create(path)?;
        let mut writer = BufWriter::new(&mut file);

        let bytes_written = ivf_builder.posting_lists_mut().write(&mut writer)?;
        Ok(bytes_written)
    }

    fn write_header(&self, header: &Header, writer: &mut BufWriter<&mut File>) -> Result<usize> {
        let version_value: u8 = match header.version {
            Version::V0 => 0,
        };
        let mut written = 0;
        written += wrap_write(writer, &version_value.to_le_bytes())?;
        written += wrap_write(writer, &header.num_features.to_le_bytes())?;
        written += wrap_write(writer, &header.num_clusters.to_le_bytes())?;
        written += wrap_write(writer, &header.num_vectors.to_le_bytes())?;
        written += wrap_write(writer, &header.doc_id_mapping_len.to_le_bytes())?;
        written += wrap_write(writer, &header.centroids_len.to_le_bytes())?;
        written += wrap_write(writer, &header.posting_lists_len.to_le_bytes())?;
        Ok(written)
    }

    /// Combine all individual files into one final index file. Keep vectors file separate.
    fn combine_files(&self, header: &Header) -> Result<usize> {
        let doc_id_mapping_path = format!("{}/doc_id_mapping", self.base_directory);
        let centroids_path = format!("{}/centroids", self.base_directory);
        let posting_lists_path = format!("{}/posting_lists", self.base_directory);

        let combined_path = format!("{}/index", self.base_directory);
        let mut combined_file = File::create(combined_path)?;
        let mut combined_buffer_writer = BufWriter::new(&mut combined_file);

        let mut written = self
            .write_header(header, &mut combined_buffer_writer)
            .context("Failed to write header")?;

        // Compute padding for alignment to 8 bytes
        written += Self::write_pad(written, &mut combined_buffer_writer, 8)?;
        written += append_file_to_writer(&doc_id_mapping_path, &mut combined_buffer_writer)?;

        // No need for padding, doc_id_mapping is always 8-byte aligned
        written += append_file_to_writer(&centroids_path, &mut combined_buffer_writer)?;

        // Pad again in case num_features and num_clusters are both odd
        written += Self::write_pad(written, &mut combined_buffer_writer, 8)?;
        written += append_file_to_writer(&posting_lists_path, &mut combined_buffer_writer)?;

        combined_buffer_writer
            .flush()
            .context("Failed to flush combined buffer")?;

        remove_file(format!("{}/doc_id_mapping", self.base_directory))?;
        remove_file(format!("{}/centroids", self.base_directory))?;
        remove_file(format!("{}/posting_lists", self.base_directory))?;

        Ok(written)
    }

    // Write padding for alignment to `alignment` bytes.
    fn write_pad(
        written: usize,
        writer: &mut BufWriter<&mut File>,
        alignment: usize,
    ) -> Result<usize> {
        let mut padded = 0;
        let padding = alignment - (written % alignment);
        if padding != alignment {
            let padding_buffer = vec![0; padding];
            padded += wrap_write(writer, &padding_buffer)?;
        }
        Ok(padded)
    }
}

// Test
#[cfg(test)]
mod tests {
    use std::fs;
    use std::io::Read;
    use std::path::Path;

    use byteorder::{LittleEndian, ReadBytesExt};
    use quantization::no_op::NoQuantizer;
    use quantization::pq::pq::ProductQuantizer;
    use tempdir::TempDir;
    use utils::test_utils::generate_random_vector;

    use super::*;
    use crate::ivf::builder::IvfBuilderConfig;

    fn create_test_file(base_directory: &str, name: &str, content: &[u8]) -> Result<()> {
        let path = format!("{}/{}", base_directory, name);
        let mut file = File::create(path)?;
        file.write_all(content)?;
        Ok(())
    }

    #[test]
    fn test_combine_files() -> Result<()> {
        // Create a temporary directory for testing
        let temp_dir = tempdir::TempDir::new("ivf_builder_test")
            .expect("Failed to create temporary directory");
        let base_directory = temp_dir
            .path()
            .to_str()
            .expect("Failed to convert temporary directory path to string")
            .to_string();

        // Create an IvfWriter instance
        let num_features = 10;
        let quantizer = NoQuantizer::new(num_features);
        let ivf_writer = IvfWriter::new(base_directory.clone(), quantizer);

        // Create test files
        create_test_file(&base_directory, "centroids", &[5, 6, 7, 8])?;
        create_test_file(&base_directory, "posting_lists", &[9, 10, 11, 12])?;
        create_test_file(&base_directory, "doc_id_mapping", &[100, 101, 102, 103])?;

        // Create a test header
        let header = Header {
            version: Version::V0,
            num_features: num_features as u32,
            num_clusters: 5,
            num_vectors: 4,
            doc_id_mapping_len: 4,
            centroids_len: 4,
            posting_lists_len: 4,
        };

        // Call combine_files
        let bytes_written = ivf_writer.combine_files(&header)?;

        // Verify the combined file
        let combined_path = format!("{}/index", base_directory);
        let mut combined_file = File::open(combined_path)?;
        let mut combined_content = Vec::new();
        combined_file.read_to_end(&mut combined_content)?;

        // Check the total bytes written
        assert_eq!(bytes_written, combined_content.len());

        // Verify the header
        let mut expected_header = vec![
            0u8, // Version::V0
            10, 0, 0, 0, // num_features (little-endian)
            5, 0, 0, 0, // num_clusters (little-endian)
            4, 0, 0, 0, 0, 0, 0, 0, // num_vectors (little-endian)
            4, 0, 0, 0, 0, 0, 0, 0, // doc_id_mapping_len (little-endian)
            4, 0, 0, 0, 0, 0, 0, 0, // centroids_len (little-endian)
            4, 0, 0, 0, 0, 0, 0, 0, // posting_lists_len (little-endian)
        ];

        // Add padding to align to 8 bytes
        while expected_header.len() % 8 != 0 {
            expected_header.push(0);
        }

        assert_eq!(
            &combined_content[..expected_header.len()],
            expected_header.as_slice()
        );

        // Verify the content of the files
        // doc_id_mapping
        let offset = expected_header.len();
        assert_eq!(&combined_content[offset..offset + 4], [100, 101, 102, 103]);

        // centroids
        let mut next_offset = offset + 4;
        assert_eq!(
            &combined_content[next_offset..next_offset + 4],
            [5, 6, 7, 8]
        );

        // Check for padding after centroids
        next_offset += 4;
        while next_offset % 8 != 0 {
            assert_eq!(combined_content[next_offset], 0);
            next_offset += 1;
        }

        assert_eq!(
            &combined_content[next_offset..next_offset + 4],
            [9, 10, 11, 12]
        );

        Ok(())
    }

    #[test]
    fn test_write_pad() {
        let temp_dir = TempDir::new("test_write_pad").unwrap();
        let path = temp_dir.path().join("test_pad");
        let mut file = File::create(&path).unwrap();
        let mut writer = BufWriter::new(&mut file);

        // Write some initial data
        let initial_size = writer.write(&[1, 2, 3]).unwrap() as usize;

        // Pad to 8-byte alignment
        let padding_written =
            IvfWriter::<NoQuantizer>::write_pad(initial_size, &mut writer, 8).unwrap();

        assert_eq!(padding_written, 5); // 3 bytes written, so 5 bytes of padding needed

        writer.flush().unwrap();

        // Check file size
        let metadata = fs::metadata(&path).unwrap();
        assert_eq!(metadata.len(), 8); // 3 bytes of data + 5 bytes of padding
    }

    #[test]
    fn test_quantize_and_write_vectors() {
        // Setup
        let temp_dir = TempDir::new("test_quantize_and_write_vectors")
            .expect("Failed to create temporary directory");
        let base_directory = temp_dir
            .path()
            .to_str()
            .expect("Failed to convert temporary directory path to string")
            .to_string();
        let num_clusters = 1;
        let num_vectors = 2;
        let num_features = 3;
        let subvector_dimension = 1;
        let file_size = 4096;

        let codebook = vec![1.5, 4.5, 2.3, 5.3, 3.1, 6.1];
        let quantizer =
            ProductQuantizer::new(3, 1, subvector_dimension, codebook, base_directory.clone())
                .expect("Can't create product quantizer");
        let ivf_writer = IvfWriter::new(base_directory.clone(), quantizer);

        let mut ivf_builder = IvfBuilder::new(IvfBuilderConfig {
            max_iteration: 1000,
            batch_size: 4,
            num_clusters,
            num_data_points: num_vectors,
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

        ivf_builder
            .add_vector(0, &vec![1.0, 2.0, 3.0])
            .expect("Vector should be added");
        ivf_builder
            .add_vector(1, &vec![4.0, 5.0, 6.0])
            .expect("Vector should be added");

        // Act
        let result = ivf_writer.quantize_and_write_vectors(&ivf_builder);
        assert!(result.is_ok());

        let bytes_written = result.unwrap();
        assert_eq!(bytes_written, 14); // 8 (number of vectors) + 6 (3 bytes each vector)

        let vectors_path = format!("{}/vectors", base_directory);
        assert!(Path::new(&vectors_path).exists());

        // Read the quantized vectors file
        let file = fs::File::open(vectors_path).expect("Failed to open vectors file");
        let mut reader = std::io::BufReader::new(file);
        let mut buffer = Vec::new();
        reader
            .read_to_end(&mut buffer)
            .expect("Failed to read vectors file");

        let expected_header = num_vectors.to_le_bytes();
        assert_eq!(&buffer[0..8], &expected_header);

        // Verify the contents of the quantized vectors
        let expected_quantized_vector_length = num_features / subvector_dimension as usize;
        let data_start = 8; // Skip the header
        let data_end = buffer.len();
        assert_eq!(
            (data_end - data_start) % expected_quantized_vector_length,
            0
        );

        let num_vectors_written = (data_end - data_start) / expected_quantized_vector_length;
        assert_eq!(num_vectors_written, num_vectors);

        // Check each quantized vector
        let expected_quantized_vectors = vec![vec![0u8, 0u8, 0u8], vec![1u8, 1u8, 1u8]];

        for i in 0..num_vectors_written {
            let start = data_start + i * expected_quantized_vector_length;
            let end = start + expected_quantized_vector_length;
            let quantized_vector = &buffer[start..end];

            assert_eq!(
                quantized_vector, &expected_quantized_vectors[i],
                "Quantized vector {} does not match expected",
                i
            );
        }
    }

    #[test]
    fn test_ivf_writer_write() {
        let temp_dir =
            TempDir::new("test_ivf_writer_write").expect("Failed to create temporary directory");
        let base_directory = temp_dir
            .path()
            .to_str()
            .expect("Failed to convert temporary directory path to string")
            .to_string();
        let num_clusters = 10;
        let num_vectors = 1000;
        let num_features = 4;
        let file_size = 4096;
        let quantizer = NoQuantizer::new(num_features);
        let writer = IvfWriter::new(base_directory.clone(), quantizer);

        let mut builder = IvfBuilder::new(IvfBuilderConfig {
            max_iteration: 1000,
            batch_size: 4,
            num_clusters,
            num_data_points: num_vectors,
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
        let mut original_vectors = Vec::new();
        for i in 0..num_vectors {
            let vector = generate_random_vector(num_features);
            original_vectors.push(vector.clone());
            builder
                .add_vector((i + 100) as u64, &vector)
                .expect("Vector should be added");
        }
        assert_eq!(builder.doc_id_mapping().len(), 1000);

        assert!(builder.build().is_ok());
        assert!(writer.write(&mut builder, false).is_ok());

        // Check if files were created and removed correctly
        assert!(fs::metadata(format!("{}/vectors", base_directory)).is_ok());
        assert!(fs::metadata(format!("{}/index", base_directory)).is_ok());
        assert!(fs::metadata(format!("{}/doc_id_mapping", base_directory)).is_err());
        assert!(fs::metadata(format!("{}/centroids", base_directory)).is_err());
        assert!(fs::metadata(format!("{}/posting_lists", base_directory)).is_err());

        // Verify vectors file content
        let vectors_file =
            File::open(format!("{}/vectors", base_directory)).expect("Failed to open vectors file");
        let mut vectors_reader = std::io::BufReader::new(vectors_file);

        let stored_num_vectors = vectors_reader
            .read_u64::<LittleEndian>()
            .expect("Failed to read number of vectors");
        assert_eq!(stored_num_vectors, num_vectors as u64);

        for original_vector in original_vectors {
            for &original_value in &original_vector {
                let stored_value = vectors_reader
                    .read_f32::<LittleEndian>()
                    .expect("Failed to read vector value");
                assert!((original_value - stored_value).abs() < f32::EPSILON);
            }
        }

        // Verify index file content
        let mut index_file =
            File::open(format!("{}/index", base_directory)).expect("Failed to open index file");
        let mut index_reader = std::io::BufReader::new(&mut index_file);

        // Read and verify header
        let version = index_reader.read_u8().expect("Failed to read version");
        assert_eq!(version, 0); // Version::V0

        let stored_num_features = index_reader
            .read_u32::<LittleEndian>()
            .expect("Failed to read num_features");
        assert_eq!(stored_num_features, num_features as u32);

        let stored_num_clusters = index_reader
            .read_u32::<LittleEndian>()
            .expect("Failed to read num_clusters");
        assert_eq!(stored_num_clusters, num_clusters as u32);

        let stored_num_vectors = index_reader
            .read_u64::<LittleEndian>()
            .expect("Failed to read num_vectors");
        assert_eq!(stored_num_vectors, num_vectors as u64);

        let doc_id_mapping_len = index_reader
            .read_u64::<LittleEndian>()
            .expect("Failed to read doc_id_mapping_len");
        assert_eq!(
            doc_id_mapping_len,
            (std::mem::size_of::<u64>() * (num_vectors + 1)) as u64
        );

        let centroids_len = index_reader
            .read_u64::<LittleEndian>()
            .expect("Failed to read centroids_len");
        let posting_lists_len = index_reader
            .read_u64::<LittleEndian>()
            .expect("Failed to read posting_lists_len");

        // Verify file size
        let file_size = index_file
            .metadata()
            .expect("Failed to get file metadata")
            .len();
        assert_eq!(
            file_size,
            41 + 7 + doc_id_mapping_len + centroids_len + posting_lists_len
        ); // 41 bytes for header + 7 padding
    }
}
