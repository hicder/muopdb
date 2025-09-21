use std::fs::File;

use anyhow::Result;
use byteorder::{ByteOrder, LittleEndian};
use memmap2::Mmap;
use quantization::quantization::Quantizer;

use crate::hnsw::index::Hnsw;
use crate::hnsw::writer::{Header, Version};
use crate::vector::fixed_file::FixedFileVectorStorage;
use crate::vector::VectorStorage;

pub struct HnswReader {
    base_directory: String,
    index_offset: usize,
    vector_offset: usize,
}

impl HnswReader {
    pub fn new(base_directory: String) -> Self {
        Self {
            base_directory,
            index_offset: 0,
            vector_offset: 0,
        }
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

    pub fn read<Q: Quantizer>(&self) -> Result<Hnsw<Q>> {
        let backing_file = File::open(format!("{}/hnsw/index", self.base_directory))?;
        let mmap = unsafe { Mmap::map(&backing_file) }?;

        let (header, offset) = self.read_header(&mmap);

        let vector_storage_path = format!("{}/hnsw/vector_storage", self.base_directory);
        let vector_storage = Box::new(VectorStorage::FixedLocalFileBacked(
            FixedFileVectorStorage::<Q::QuantizedT>::new_with_offset(
                vector_storage_path,
                header.quantized_dimension as usize,
                self.vector_offset,
            )?,
        ));
        let edges_padding = (4 - (offset % 4)) % 4;
        let edges_offset = offset + edges_padding;
        let points_offset = edges_offset + header.edges_len as usize;

        let edge_offsets_padding = (8 - ((points_offset + header.points_len as usize) % 8)) % 8;
        let edge_offsets_offset = points_offset + header.points_len as usize + edge_offsets_padding;
        let level_offsets_offset = edge_offsets_offset + header.edge_offsets_len as usize;

        let doc_id_mapping_padding =
            (16 - ((level_offsets_offset + header.level_offsets_len as usize) % 16)) % 16;
        let doc_id_mapping_offset =
            level_offsets_offset + header.level_offsets_len as usize + doc_id_mapping_padding;

        Ok(Hnsw::new(
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
        ))
    }

    /// Read the header from the mmap and return the header and the offset of data page
    pub fn read_header(&self, buffer: &[u8]) -> (Header, usize) {
        let mut offset = self.index_offset;
        let version = match buffer[offset] {
            0 => Version::V0,
            default => panic!("Unknown version: {}", default),
        };

        offset += 1;
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

    use quantization::noq::noq::NoQuantizer;
    use quantization::pq::pq::{ProductQuantizer, ProductQuantizerConfig};
    use quantization::pq::pq_builder::{ProductQuantizerBuilder, ProductQuantizerBuilderConfig};
    use quantization::quantization::WritableQuantizer;
    use utils::distance::l2::L2DistanceCalculator;
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
        let mut pq_builder = ProductQuantizerBuilder::new(pq_config, pq_builder_config);

        for datapoint in datapoints.iter().take(1000) {
            pq_builder.add(datapoint.clone());
        }
        let pq = pq_builder.build(base_directory.clone()).unwrap();
        assert!(pq.write_to_directory(&pq_dir).is_ok());

        // Create a HNSW Builder
        let vector_dir = format!("{}/vectors", base_directory);
        fs::create_dir_all(vector_dir.clone()).unwrap();
        let mut hnsw_builder = HnswBuilder::<ProductQuantizer<L2DistanceCalculator>>::new(
            10, 128, 20, 1024, 4096, 16, pq, vector_dir,
        );
        for (i, datapoint) in datapoints.iter().enumerate() {
            hnsw_builder.insert(i as u128, datapoint).unwrap();
        }

        let hnsw_dir = format!("{}/hnsw", base_directory);
        fs::create_dir_all(hnsw_dir.clone()).unwrap();
        let writer = HnswWriter::new(hnsw_dir);
        assert!(writer.write(&mut hnsw_builder, false).is_ok());

        // Read from file
        let reader = HnswReader::new(base_directory.clone());
        let hnsw = reader
            .read::<ProductQuantizer<L2DistanceCalculator>>()
            .unwrap();
        assert_eq!(49, hnsw.get_data_offset());
        assert_eq!(16, hnsw.get_header().quantized_dimension);
    }

    #[test]
    fn test_read_no_op_quantizer() {
        let temp_dir = tempdir::TempDir::new("product_quantizer_test").unwrap();
        let base_directory = temp_dir.path().to_str().unwrap().to_string();
        let vector_dir = format!("{}/vectors", base_directory);
        fs::create_dir_all(vector_dir.clone()).unwrap();
        let datapoints: Vec<Vec<f32>> = (0..10000).map(|_| generate_random_vector(128)).collect();

        // quantizer
        let quantizer = NoQuantizer::<L2DistanceCalculator>::new(128);
        let quantizer_dir = format!("{}/quantizer", base_directory);
        fs::create_dir_all(quantizer_dir.clone()).unwrap();
        assert!(quantizer.write_to_directory(&quantizer_dir).is_ok());

        let mut hnsw_builder =
            HnswBuilder::new(10, 128, 20, 1024, 4096, 128, quantizer, vector_dir);
        for (i, datapoint) in datapoints.iter().enumerate() {
            hnsw_builder.insert(i as u128, datapoint).unwrap();
        }

        let hnsw_dir = format!("{}/hnsw", base_directory);
        fs::create_dir_all(hnsw_dir.clone()).unwrap();
        let writer = HnswWriter::new(hnsw_dir);
        assert!(writer.write(&mut hnsw_builder, false).is_ok());

        // Read from file
        let reader = HnswReader::new(base_directory.clone());
        let hnsw = reader.read::<NoQuantizer<L2DistanceCalculator>>().unwrap();
        assert_eq!(49, hnsw.get_data_offset());
        assert_eq!(128, hnsw.get_header().quantized_dimension);
    }
}
