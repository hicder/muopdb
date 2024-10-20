use std::fs::File;

use byteorder::{ByteOrder, LittleEndian};
use memmap2::Mmap;

use crate::hnsw::index::Hnsw;
use crate::hnsw::writer::{Header, Version};

pub struct HnswReader {
    base_directory: String,
}

impl HnswReader {
    pub fn new(base_directory: String) -> Self {
        Self { base_directory }
    }

    pub fn read(&self) -> Hnsw {
        let backing_file = File::open(format!("{}/index", self.base_directory)).unwrap();
        let mmap = unsafe { Mmap::map(&backing_file).unwrap() };

        let (header, offset) = self.read_header(&mmap);
        let edges_padding = (4 - (offset % 4)) % 4;
        let edges_offset = offset + edges_padding as usize;
        let points_offset = edges_offset + header.edges_len as usize;

        let edge_offsets_padding = (8 - ((points_offset + header.points_len as usize) % 8)) % 8;
        let edge_offsets_offset =
            points_offset + header.points_len as usize + edge_offsets_padding as usize;
        let level_offsets_offset = edge_offsets_offset + header.edge_offsets_len as usize;

        Hnsw::new(
            backing_file,
            mmap,
            header,
            offset,
            edges_offset,
            points_offset,
            edge_offsets_offset,
            level_offsets_offset,
            self.base_directory.clone(),
        )
    }

    /// Read the header from the mmap and return the header and the offset of data page
    pub fn read_header(&self, buffer: &[u8]) -> (Header, usize) {
        let version = match buffer[0] {
            0 => Version::V0,
            default => panic!("Unknown version: {}", default),
        };

        let mut offset = 1;
        let num_layers = LittleEndian::read_u32(&buffer[offset..]);
        offset += 4;
        let edges_len = LittleEndian::read_u64(&buffer[offset..]);
        offset += 8;
        let points_len = LittleEndian::read_u64(&buffer[offset..]);
        offset += 8;
        let edge_offsets_len = LittleEndian::read_u64(&buffer[offset..]);
        offset += 8;
        let level_offsets_len = LittleEndian::read_u64(&buffer[offset..]);
        offset += 8;

        (
            Header {
                version,
                num_layers,
                level_offsets_len,
                edges_len,
                points_len,
                edge_offsets_len,
            },
            offset,
        )
    }
}

// Test
#[cfg(test)]
mod tests {
    use quantization::pq::{ProductQuantizerConfig, ProductQuantizerWriter};
    use quantization::pq_builder::{ProductQuantizerBuilder, ProductQuantizerBuilderConfig};
    use utils::test_utils::generate_random_vector;

    use super::*;
    use crate::hnsw::builder::HnswBuilder;
    use crate::hnsw::writer::HnswWriter;

    #[test]
    fn test_read_header() {
        // Generate 10000 vectors of f32, dimension 128
        let datapoints: Vec<Vec<f32>> = (0..10000).map(|_| generate_random_vector(128)).collect();

        // Create a temporary directory
        let temp_dir = tempdir::TempDir::new("product_quantizer_test").unwrap();
        let base_directory = temp_dir.path().to_str().unwrap().to_string();
        let pq_config = ProductQuantizerConfig {
            dimension: 128,
            subvector_dimension: 8,
            num_bits: 8,
            base_directory: base_directory.clone(),
            codebook_name: "codebook".to_string(),
        };

        let pq_builder_config = ProductQuantizerBuilderConfig {
            max_iteration: 1000,
            batch_size: 4,
        };

        // Train a product quantizer
        let pq_writer = ProductQuantizerWriter::new(pq_config.base_directory.clone());
        let mut pq_builder = ProductQuantizerBuilder::new(pq_config, pq_builder_config);

        for i in 0..1000 {
            pq_builder.add(datapoints[i].clone());
        }
        let pq = pq_builder.build().unwrap();
        pq_writer.write(&pq).unwrap();

        // Create a HNSW Builder
        let mut hnsw_builder = HnswBuilder::new(10, 128, 20, Box::new(pq));
        for i in 0..datapoints.len() {
            hnsw_builder.insert(i as u64, &datapoints[i]);
        }

        let writer = HnswWriter::new(base_directory.clone());
        match writer.write(&hnsw_builder) {
            Ok(()) => {
                assert!(true);
            }
            Err(_) => {
                assert!(false);
            }
        }

        // Read from file
        let reader = HnswReader::new(base_directory.clone());
        let hnsw = reader.read();
        assert_eq!(37, hnsw.get_data_offset());
    }
}
