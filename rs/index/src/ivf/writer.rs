use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};

use anyhow::{anyhow, Context, Result};
use utils::io::wrap_write;

use crate::ivf::index::Ivf;

pub struct IvfWriter {
    base_directory: String,
}

#[derive(PartialEq, Debug)]
pub enum Version {
    V0,
}

impl IvfWriter {
    pub fn new(base_directory: String) -> Self {
        Self { base_directory }
    }

    pub fn write(&self, ivf: &Ivf) -> Result<()> {
        // Write data
        self.write_data(&ivf.dataset, Version::V0)
            .context("failed to write data")?;

        // Write centroids
        self.write_centroids(
            &ivf.dataset,
            &ivf.centroids,
            &ivf.inverted_lists,
            Version::V0,
        )
        .context("failed to write centroids")?;

        // Write inverted lists
        self.write_inverted_lists(&ivf.inverted_lists, ivf.num_probes as u32, Version::V0)
            .context("failed to write inverted lists")?;

        Ok(())
    }

    fn write_data(&self, dataset: &Vec<Vec<f32>>, version: Version) -> Result<()> {
        let path = format!("{}/data", self.base_directory);
        let mut file = File::create(path)?;
        let mut writer = BufWriter::new(&mut file);

        let version_value: u8 = match version {
            Version::V0 => 0,
        };
        let dataset_size = dataset.len() as u64;
        let vector_dimension = dataset[0].len() as u32;

        // Write header.
        let mut written = 0;
        written += wrap_write(&mut writer, &version_value.to_le_bytes())?;
        written += wrap_write(&mut writer, &dataset_size.to_le_bytes())?;
        written += wrap_write(&mut writer, &vector_dimension.to_le_bytes())?;
        if written != 1 + 8 + 4 {
            return Err(anyhow!(
                "Expected to write 13 bytes as data header, but wrote {}",
                written,
            ));
        }

        // Pad to 8-byte alignment.
        Self::write_pad(written, &mut writer, 8)?;

        // Write the actual dataset.
        for vector in dataset.iter() {
            for elem in vector.iter() {
                let bytes_written = wrap_write(&mut writer, &elem.to_le_bytes())?;
                if bytes_written != 4 {
                    return Err(anyhow!(
                        "Expected to write 4 bytes, but wrote {}",
                        bytes_written,
                    ));
                }
            }
        }
        writer.flush()?;

        Ok(())
    }

    fn write_centroids(
        &self,
        dataset: &Vec<Vec<f32>>,
        centroids: &Vec<Vec<f32>>,
        inverted_lists: &HashMap<usize, Vec<usize>>,
        version: Version,
    ) -> Result<()> {
        let path = format!("{}/centroids", self.base_directory);
        let mut file = File::create(path)?;
        let mut writer = BufWriter::new(&mut file);

        let version_value: u8 = match version {
            Version::V0 => 0,
        };
        let num_clusters = centroids.len() as u32;
        let vector_dimension = dataset[0].len() as u32;

        // Write header.
        let mut written = 0;
        written += wrap_write(&mut writer, &version_value.to_le_bytes())?;
        written += wrap_write(&mut writer, &num_clusters.to_le_bytes())?;
        written += wrap_write(&mut writer, &vector_dimension.to_le_bytes())?;
        if written != 1 + 4 + 4 {
            return Err(anyhow!(
                "Expected to write 9 bytes as centroid header, but wrote {}",
                written,
            ));
        }

        // Pad to 8-byte alignment.
        Self::write_pad(written, &mut writer, 8)?;

        // Write the centroids with all the vectors in the same cluster.
        for (idx, centroid) in centroids.iter().enumerate() {
            // Write the centroid vector.
            for elem in centroid.iter() {
                let bytes_written = wrap_write(&mut writer, &elem.to_le_bytes())?;
                if bytes_written != 4 {
                    return Err(anyhow!(
                        "Expected to write 4 bytes, but wrote {}",
                        bytes_written,
                    ));
                }
            }
            // Write the vectors from the same cluster.
            let vector_indices = &inverted_lists[&idx];
            let cluster_size = vector_indices.len() as u64;
            wrap_write(&mut writer, &cluster_size.to_le_bytes())?;
            for vector_idx in vector_indices.iter() {
                for elem in dataset[*vector_idx].iter() {
                    let bytes_written = wrap_write(&mut writer, &elem.to_le_bytes())?;
                    if bytes_written != 4 {
                        return Err(anyhow!(
                            "Expected to write 4 bytes, but wrote {}",
                            bytes_written,
                        ));
                    }
                }
            }
        }
        writer.flush()?;

        Ok(())
    }

    fn write_inverted_lists(
        &self,
        inverted_lists: &HashMap<usize, Vec<usize>>,
        num_probes: u32,
        version: Version,
    ) -> Result<()> {
        // TODO(tyb0807): sort by doc_id when it's used.
        let path = format!("{}/inverted_lists", self.base_directory);
        let mut file = File::create(path)?;
        let mut writer = BufWriter::new(&mut file);

        let version_value: u8 = match version {
            Version::V0 => 0,
        };
        let num_inverted_list_indices = inverted_lists.len() as u32;

        // Write header.
        let mut written = 0;
        written += wrap_write(&mut writer, &version_value.to_le_bytes())?;
        written += wrap_write(&mut writer, &num_probes.to_le_bytes())?;
        written += wrap_write(&mut writer, &num_inverted_list_indices.to_le_bytes())?;
        if written != 1 + 4 + 4 {
            return Err(anyhow!(
                "Expected to write 13 bytes as data header, but wrote {}",
                written,
            ));
        }

        // Pad to 8-byte alignment.
        Self::write_pad(written, &mut writer, 8)?;

        // Write interted lists
        for (centroid_idx, vector_indices) in inverted_lists {
            let ci = *centroid_idx as u32;
            wrap_write(&mut writer, &ci.to_le_bytes())?;
            // Write the number of vectors in this centroid.
            let vectors_in_centroid = vector_indices.len() as u64;
            wrap_write(&mut writer, &vectors_in_centroid.to_le_bytes())?;
            for vector_idx in vector_indices.iter() {
                let vi = *vector_idx as u64;
                wrap_write(&mut writer, &vi.to_le_bytes())?;
            }
        }
        writer.flush()?;

        Ok(())
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

    use tempdir::TempDir;

    use super::*;

    fn create_test_ivf() -> Ivf {
        let dataset = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let centroids = vec![vec![1.5, 2.5, 3.5], vec![5.5, 6.5, 7.5]];
        let mut inverted_lists = HashMap::new();
        inverted_lists.insert(0, vec![0]);
        inverted_lists.insert(1, vec![1, 2]);

        Ivf {
            dataset,
            num_clusters: 1,
            centroids,
            inverted_lists,
            num_probes: 2,
        }
    }

    #[test]
    fn test_ivf_writer_write() {
        let temp_dir = TempDir::new("test_ivf_writer_write").unwrap();
        let base_directory = temp_dir.path().to_str().unwrap().to_string();
        let writer = IvfWriter::new(base_directory.clone());
        let ivf = create_test_ivf();

        assert!(writer.write(&ivf).is_ok());

        // Check if files were created
        assert!(fs::metadata(format!("{}/data", base_directory)).is_ok());
        assert!(fs::metadata(format!("{}/centroids", base_directory)).is_ok());
        assert!(fs::metadata(format!("{}/inverted_lists", base_directory)).is_ok());
    }

    #[test]
    fn test_write_data() {
        let temp_dir = TempDir::new("test_write_data").unwrap();
        let base_directory = temp_dir.path().to_str().unwrap().to_string();
        let writer = IvfWriter::new(base_directory.clone());
        let dataset = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];

        assert!(writer.write_data(&dataset, Version::V0).is_ok());
        // Calculate expected file size
        let expected_size = 1 + 8 + 4 + // Header: version (1 byte) + dataset_size (8 bytes) + vector_dimension (4 bytes)
                            (8 - (13 % 8)) + // Padding to 8-byte alignment
                            (4 * 3 * 2); // Data: 4 bytes per float, 3 floats per vector, 2 vectors

        // Check if data file was created and has correct size
        let metadata = fs::metadata(format!("{}/data", base_directory)).unwrap();
        assert_eq!(
            metadata.len(),
            expected_size as u64,
            "File size doesn't match expected size"
        );
    }

    #[test]
    fn test_write_centroids() {
        let temp_dir = TempDir::new("test_write_centroids").unwrap();
        let base_directory = temp_dir.path().to_str().unwrap().to_string();
        let writer = IvfWriter::new(base_directory.clone());
        let dataset = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let centroids = vec![vec![1.5, 2.5, 3.5]];
        let mut inverted_lists = HashMap::new();
        inverted_lists.insert(0, vec![0, 1]);

        assert!(writer
            .write_centroids(&dataset, &centroids, &inverted_lists, Version::V0)
            .is_ok());
        // Calculate expected file size
        let expected_size = 1 + 4 + 4 + // Header: version (1 byte) + num_clusters (4 bytes) + vector_dimension (4 bytes)
                            (8 - (9 % 8)) + // Padding to 8-byte alignment
                            (4 * 3) + // Centroid vector: 4 bytes per float, 3 floats
                            8 + // Cluster size (u64)
                            (4 * 3 * 2); // Cluster vectors: 4 bytes per float, 3 floats per vector, 2 vectors

        // Check if centroids file was created and has correct size
        let metadata = fs::metadata(format!("{}/centroids", base_directory)).unwrap();
        assert_eq!(
            metadata.len(),
            expected_size as u64,
            "File size doesn't match expected size"
        );
    }

    #[test]
    fn test_write_inverted_lists() {
        let temp_dir = TempDir::new("test_write_inverted_lists").unwrap();
        let base_directory = temp_dir.path().to_str().unwrap().to_string();
        let writer = IvfWriter::new(base_directory.clone());
        let mut inverted_lists = HashMap::new();
        inverted_lists.insert(0, vec![0, 1]);
        inverted_lists.insert(1, vec![2, 3]);

        assert!(writer
            .write_inverted_lists(&inverted_lists, 2, Version::V0)
            .is_ok());

        // Calculate expected file size
        let expected_size = 1 + 4 + 4 + // Header: version (1 byte) + num_probes (4 bytes) + num_inverted_list_indices (4 bytes)
                            (8 - (9 % 8)) + // Padding to 8-byte alignment
                            (4 + 8 + 8 * 2) * 2; // For each centroid: centroid_idx (4 bytes) + vectors_in_centroid (8 bytes) + vector_indices (8 bytes each)

        // Check if inverted_lists file was created and has correct size
        let metadata = fs::metadata(format!("{}/inverted_lists", base_directory)).unwrap();
        assert_eq!(
            metadata.len(),
            expected_size as u64,
            "File size doesn't match expected size"
        );
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
}
