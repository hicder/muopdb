use std::fs::File;

use byteorder::{ByteOrder, LittleEndian};
use memmap2::Mmap;

use crate::hnsw::index::Hnsw;
use crate::hnsw::writer::{Header, Version};
use crate::vector::fixed_file::FixedFileVectorStorage;

pub struct HnswReader {
    base_directory: String,
}

impl HnswReader {
    pub fn new(base_directory: String) -> Self {
        Self { base_directory }
    }

    pub fn read(&self) -> Hnsw {
        let backing_file = File::open(format!("{}/hnsw/index", self.base_directory)).unwrap();
        let mmap = unsafe { Mmap::map(&backing_file).unwrap() };

        let (header, offset) = self.read_header(&mmap);

        let vector_storage_path = format!("{}/hnsw/vector_storage", self.base_directory);
        let vector_storage = FixedFileVectorStorage::<u8>::new(
            vector_storage_path,
            header.quantized_dimension as usize,
        )
        .unwrap();
        let edges_padding = (4 - (offset % 4)) % 4;
        let edges_offset = offset + edges_padding as usize;
        let points_offset = edges_offset + header.edges_len as usize;

        let edge_offsets_padding = (8 - ((points_offset + header.points_len as usize) % 8)) % 8;
        let edge_offsets_offset =
            points_offset + header.points_len as usize + edge_offsets_padding as usize;
        let level_offsets_offset = edge_offsets_offset + header.edge_offsets_len as usize;
        let doc_id_mapping_offset = level_offsets_offset + header.level_offsets_len as usize;

        Hnsw::new(
            backing_file,
            vector_storage,
            header,
            offset,
            edges_offset,
            points_offset,
            edge_offsets_offset,
            level_offsets_offset,
            doc_id_mapping_offset,
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
        let quantized_dimension = LittleEndian::read_u32(&buffer[offset..]);
        offset += 4;

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
        let doc_id_mapping_len = LittleEndian::read_u64(&buffer[offset..]);
        offset += 8;

        (
            Header {
                version,
                quantized_dimension,
                num_layers,
                level_offsets_len,
                edges_len,
                points_len,
                edge_offsets_len,
                doc_id_mapping_len,
            },
            offset,
        )
    }
}

// Test
#[cfg(test)]
mod tests {
    use std::fs;

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
        let pq_dir = format!("{}/quantizer", base_directory);
        fs::create_dir_all(pq_dir.clone()).unwrap();
        let pq_config = ProductQuantizerConfig {
            dimension: 128,
            subvector_dimension: 8,
            num_bits: 8,
        };

        let pq_builder_config = ProductQuantizerBuilderConfig {
            max_iteration: 1000,
            batch_size: 4,
        };

        // Train a product quantizer
        let pq_writer = ProductQuantizerWriter::new(pq_dir);
        let mut pq_builder = ProductQuantizerBuilder::new(pq_config, pq_builder_config);

        for i in 0..1000 {
            pq_builder.add(datapoints[i].clone());
        }
        let pq = pq_builder.build(base_directory.clone()).unwrap();
        pq_writer.write(&pq).unwrap();

        // Create a HNSW Builder
        let vector_dir = format!("{}/vectors", base_directory);
        fs::create_dir_all(vector_dir.clone()).unwrap();
        let mut hnsw_builder =
            HnswBuilder::new(10, 128, 20, 1024, 4096, 16, Box::new(pq), vector_dir);
        for i in 0..datapoints.len() {
            hnsw_builder.insert(i as u64, &datapoints[i]).unwrap();
        }

        let hnsw_dir = format!("{}/hnsw", base_directory);
        fs::create_dir_all(hnsw_dir.clone()).unwrap();
        let writer = HnswWriter::new(hnsw_dir);
        match writer.write(&mut hnsw_builder, false) {
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
        assert_eq!(49, hnsw.get_data_offset());
        assert_eq!(16, hnsw.get_header().quantized_dimension);
    }
}
