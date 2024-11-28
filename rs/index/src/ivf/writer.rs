use std::fs::{remove_file, File};
use std::io::{BufWriter, Write};

use anyhow::{anyhow, Context, Result};
use utils::io::{append_file_to_writer, wrap_write};

use crate::ivf::builder::IvfBuilder;
use crate::posting_list::combined_file::{Header, Version};

pub struct IvfWriter {
    base_directory: String,
}

impl IvfWriter {
    pub fn new(base_directory: String) -> Self {
        Self { base_directory }
    }

    pub fn write(&self, ivf_builder: &mut IvfBuilder) -> Result<()> {
        let num_features = ivf_builder.config().num_features;
        let num_clusters = ivf_builder.config().num_clusters;
        let num_vectors = ivf_builder.vectors().len();

        // Write vectors
        let vectors_len = self
            .write_vectors(ivf_builder)
            .context("Failed to write vectors")?;
        let expected_bytes_written =
            std::mem::size_of::<u64>() + std::mem::size_of::<f32>() * num_features * num_vectors;
        if vectors_len != expected_bytes_written {
            return Err(anyhow!(
                "Expected to write {} bytes in vector storage, but wrote {}",
                expected_bytes_written,
                vectors_len,
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
            centroids_len: centroids_len as u64,
            posting_lists_len: posting_lists_len as u64,
        };

        self.combine_files(&header)?;
        Ok(())
    }

    fn write_vectors(&self, ivf_builder: &IvfBuilder) -> Result<usize> {
        let path = format!("{}/vectors", self.base_directory);
        let mut file = File::create(path)?;
        let mut writer = BufWriter::new(&mut file);

        let bytes_written = ivf_builder.vectors().write(&mut writer)?;
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
        written += wrap_write(writer, &header.centroids_len.to_le_bytes())?;
        written += wrap_write(writer, &header.posting_lists_len.to_le_bytes())?;
        Ok(written)
    }

    /// Combine all individual files into one final index file. Keep vectors file separate.
    fn combine_files(&self, header: &Header) -> Result<usize> {
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
        written += append_file_to_writer(&centroids_path, &mut combined_buffer_writer)?;

        // Pad again in case num_features and num_clusters are both odd
        written += Self::write_pad(written, &mut combined_buffer_writer, 8)?;
        written += append_file_to_writer(&posting_lists_path, &mut combined_buffer_writer)?;

        combined_buffer_writer
            .flush()
            .context("Failed to flush combined buffer")?;

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

    use byteorder::{LittleEndian, ReadBytesExt};
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
        let ivf_writer = IvfWriter::new(base_directory.clone());

        // Create test files
        create_test_file(&base_directory, "centroids", &[5, 6, 7, 8])?;
        create_test_file(&base_directory, "posting_lists", &[9, 10, 11, 12])?;

        // Create a test header
        let header = Header {
            version: Version::V0,
            num_features: 10,
            num_clusters: 5,
            num_vectors: 4,
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
        let offset = expected_header.len();
        assert_eq!(&combined_content[offset..offset + 4], [5, 6, 7, 8]);

        // Check for padding after centroids
        let mut next_offset = offset + 4;
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
        let padding_written = IvfWriter::write_pad(initial_size, &mut writer, 8).unwrap();

        assert_eq!(padding_written, 5); // 3 bytes written, so 5 bytes of padding needed

        writer.flush().unwrap();

        // Check file size
        let metadata = fs::metadata(&path).unwrap();
        assert_eq!(metadata.len(), 8); // 3 bytes of data + 5 bytes of padding
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
        let writer = IvfWriter::new(base_directory.clone());

        let mut builder = IvfBuilder::new(IvfBuilderConfig {
            max_iteration: 1000,
            batch_size: 4,
            num_clusters,
            num_probes: 2,
            num_data_points: num_vectors,
            max_clusters_per_vector: 1,
            base_directory: base_directory.clone(),
            memory_size: 1024,
            file_size,
            num_features,
        })
        .expect("Failed to create builder");
        // Generate 1000 vectors of f32, dimension 4
        let mut original_vectors = Vec::new();
        for _ in 0..num_vectors {
            let vector = generate_random_vector(num_features);
            original_vectors.push(vector.clone());
            builder.add_vector(vector).expect("Vector should be added");
        }

        let result = builder.build();
        assert!(result.is_ok());

        assert!(writer.write(&mut builder).is_ok());

        // Check if files were created and removed correctly
        assert!(fs::metadata(format!("{}/vectors", base_directory)).is_ok());
        assert!(fs::metadata(format!("{}/index", base_directory)).is_ok());
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
        assert_eq!(file_size, 33 + 7 + centroids_len + posting_lists_len); // 33 bytes for header + 7 padding
    }
}
