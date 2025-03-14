use std::mem::size_of;

use anyhow::{anyhow, Result};
use byteorder::{ByteOrder, LittleEndian};
use memmap2::Mmap;
use utils::mem::transmute_u8_to_slice;

const PL_METADATA_LEN: usize = 2;

#[derive(PartialEq, Debug)]
pub enum Version {
    V0,
}

#[derive(Debug)]
pub struct Header {
    pub version: Version,
    pub num_features: u32,
    pub quantized_dimension: u32,
    pub num_clusters: u32,
    pub num_vectors: u64,
    pub doc_id_mapping_len: u64,
    pub centroids_len: u64,
    pub posting_lists_and_metadata_len: u64,
}

pub struct FixedIndexFile {
    mmap: Mmap,
    header: Header,
    doc_id_mapping_offset: usize,
    centroid_offset: usize,
    posting_list_metadata_offset: usize,
}

impl FixedIndexFile {
    pub fn new_with_offset(file_path: String, offset: usize) -> Result<Self> {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .open(file_path.clone())?;
        let mmap = unsafe { Mmap::map(&file) }?;
        let (header, doc_id_mapping_offset) = Self::read_header(&mmap, offset)?;

        let centroid_offset = Self::align_to_next_boundary(
            doc_id_mapping_offset + header.doc_id_mapping_len as usize,
            8,
        );

        let posting_list_metadata_offset =
            Self::align_to_next_boundary(centroid_offset + header.centroids_len as usize, 8)
                + size_of::<u64>(); // FileBackedAppendablePostingListStorage's first u64 encodes num_clusters
        Ok(Self {
            mmap,
            header,
            doc_id_mapping_offset,
            centroid_offset,
            posting_list_metadata_offset,
        })
    }

    pub fn new(file_path: String) -> Result<Self> {
        Self::new_with_offset(file_path, 0)
    }

    /// Read the header from the mmap and return the header and the offset of data page
    pub fn read_header(buffer: &[u8], offset: usize) -> Result<(Header, usize)> {
        let mut offset = offset;
        let version = match buffer[offset] {
            0 => Version::V0,
            default => return Err(anyhow!("Unknown version: {}", default)),
        };
        offset += 1;

        let num_features = LittleEndian::read_u32(&buffer[offset..]);
        offset += 4;
        let quantized_dimension = LittleEndian::read_u32(&buffer[offset..]);
        offset += 4;
        let num_clusters = LittleEndian::read_u32(&buffer[offset..]);
        offset += 4;
        let num_vectors = LittleEndian::read_u64(&buffer[offset..]);
        offset += 8;
        let doc_id_mapping_len = LittleEndian::read_u64(&buffer[offset..]);
        offset += 8;
        let centroids_len = LittleEndian::read_u64(&buffer[offset..]);
        offset += 8;
        let posting_lists_and_metadata_len = LittleEndian::read_u64(&buffer[offset..]);
        offset += 8;

        let header = Header {
            version,
            num_features,
            quantized_dimension,
            num_clusters,
            num_vectors,
            doc_id_mapping_len,
            centroids_len,
            posting_lists_and_metadata_len,
        };

        // Align to the next 8-byte boundary
        offset = Self::align_to_next_boundary(offset, 16);

        Ok((header, offset))
    }

    fn align_to_next_boundary(current_position: usize, alignment: usize) -> usize {
        let mask = alignment - 1;
        (current_position + mask) & !mask
    }

    pub fn get_doc_id(&self, index: usize) -> Result<u128> {
        if index >= self.header.num_vectors as usize {
            return Err(anyhow!("Index out of bound"));
        }

        let start = self.doc_id_mapping_offset
            + size_of::<u128>() // Read another u128 which encodes num_vectors (when combining with
                                // doc_id_mapping storage)
            + index * size_of::<u128>();
        let slice = &self.mmap[start..start + size_of::<u128>()];
        Ok(u128::from_le_bytes(slice.try_into()?))
    }

    pub fn get_centroid(&self, index: usize) -> Result<&[f32]> {
        if index >= self.header.num_clusters as usize {
            return Err(anyhow!("Index out of bound"));
        }

        let start = self.centroid_offset
            + size_of::<u64>() // Read another u64 which encodes num_clusters (when combining with
                               // centroid storage)
            + index * self.header.num_features as usize * size_of::<f32>();
        let slice = &self.mmap[start..start + self.header.num_features as usize * size_of::<f32>()];
        Ok(transmute_u8_to_slice::<f32>(slice))
    }

    pub fn get_posting_list(&self, index: usize) -> Result<&[u8]> {
        if index >= self.header.num_clusters as usize {
            return Err(anyhow!("Index out of bound"));
        }

        let metadata_offset =
            self.posting_list_metadata_offset + index * PL_METADATA_LEN * size_of::<u64>();

        let posting_list_start_offset = self.posting_list_metadata_offset
            + self.header.num_clusters as usize * PL_METADATA_LEN * size_of::<u64>();

        let slice = &self.mmap[metadata_offset..metadata_offset + size_of::<u64>()];
        let pl_len = u64::from_le_bytes(slice.try_into()?) as usize;

        let slice = &self.mmap[metadata_offset + size_of::<u64>()
            ..metadata_offset + PL_METADATA_LEN * size_of::<u64>()];
        let pl_offset = u64::from_le_bytes(slice.try_into()?) as usize + posting_list_start_offset;

        Ok(&self.mmap[pl_offset..pl_offset + pl_len])
    }

    pub fn header(&self) -> &Header {
        &self.header
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::Write;

    use utils::mem::transmute_slice_to_u8;

    use super::*;

    #[test]
    fn test_fixed_file_posting_list_storage() {
        // Create a temporary directory for testing
        let temp_dir = tempdir::TempDir::new("fixed_file_posting_list_storage_test")
            .expect("Failed to create temporary directory");
        let base_directory = temp_dir
            .path()
            .to_str()
            .expect("Failed to convert temporary directory path to string")
            .to_string();

        let path = format!("{}/index", base_directory);
        let mut file = File::create(&path).expect("Failed to create index file");

        // Create a test header
        let mut header = vec![
            0u8, // Version::V0
            4, 0, 0, 0, // num_features (little-endian)
            4, 0, 0, 0, // quantized_dimension (little-endian)
            2, 0, 0, 0, // num_clusters (little-endian)
            4, 0, 0, 0, 0, 0, 0, 0, // num_vectors (little-endian)
            80, 0, 0, 0, 0, 0, 0, 0, // doc_id_mapping_len (little-endian)
            40, 0, 0, 0, 0, 0, 0, 0, // centroids_len (little-endian)
            9, 0, 0, 0, 0, 0, 0,
            0, // posting_lists_and_metadata_len - garbage (little-endian)
        ];

        // Add padding to align to 8 bytes
        while header.len() % 8 != 0 {
            header.push(0);
        }
        assert!(file.write_all(&header).is_ok());

        let doc_id_mapping: Vec<u128> = vec![100, 200, 300, 400];
        let num_vectors = vec![4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        assert!(file.write_all(&num_vectors).is_ok());
        assert!(file
            .write_all(transmute_slice_to_u8(&doc_id_mapping))
            .is_ok());
        // No need for padding here

        let centroids: Vec<Vec<f32>> = vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]];
        let num_clusters = vec![2, 0, 0, 0, 0, 0, 0, 0];
        assert!(file.write_all(&num_clusters).is_ok());
        assert!(file.write_all(transmute_slice_to_u8(&centroids[0])).is_ok());
        assert!(file.write_all(transmute_slice_to_u8(&centroids[1])).is_ok());
        // No need for padding here

        let posting_lists: Vec<Vec<u64>> = vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8, 9, 10]];
        // Posting list offset starts at 0 (see FileBackedAppendablePostingListStorage)
        let metadata: Vec<u64> = vec![4 * 8, 0, 6 * 8, 32];
        assert!(file.write_all(&num_clusters).is_ok());
        assert!(file.write_all(transmute_slice_to_u8(&metadata)).is_ok());
        assert!(file
            .write_all(transmute_slice_to_u8(&posting_lists[0]))
            .is_ok());
        assert!(file
            .write_all(transmute_slice_to_u8(&posting_lists[1]))
            .is_ok());

        let combined_file = FixedIndexFile::new(path)
            .expect("Failed to create centroid posting list combined file");

        assert_eq!(combined_file.header.version, Version::V0);
        assert_eq!(combined_file.header.num_features, 4);
        assert_eq!(combined_file.header.quantized_dimension, 4);
        assert_eq!(combined_file.header.num_clusters, 2);
        assert_eq!(combined_file.header.num_vectors, 4);
        assert_eq!(combined_file.header.doc_id_mapping_len, 80);
        assert_eq!(combined_file.header.centroids_len, 40);
        assert_eq!(combined_file.header.posting_lists_and_metadata_len, 9);

        assert_eq!(
            combined_file
                .get_doc_id(0)
                .expect("Failed to read doc_id_mapping"),
            doc_id_mapping[0]
        );
        assert_eq!(
            combined_file
                .get_doc_id(1)
                .expect("Failed to read doc_id_mapping"),
            doc_id_mapping[1]
        );
        assert_eq!(
            combined_file
                .get_doc_id(2)
                .expect("Failed to read doc_id_mapping"),
            doc_id_mapping[2]
        );
        assert_eq!(
            combined_file
                .get_doc_id(3)
                .expect("Failed to read doc_id_mapping"),
            doc_id_mapping[3]
        );
        assert!(combined_file.get_doc_id(4).is_err());

        assert_eq!(
            combined_file
                .get_centroid(0)
                .expect("Failed to read centroid"),
            &centroids[0]
        );
        assert_eq!(
            combined_file
                .get_centroid(1)
                .expect("Failed to read centroid"),
            &centroids[1]
        );
        assert!(combined_file.get_centroid(2).is_err());

        assert_eq!(
            transmute_u8_to_slice::<u64>(
                combined_file
                    .get_posting_list(0)
                    .expect("Failed to read posting_list")
            ),
            posting_lists[0]
        );
        assert_eq!(
            transmute_u8_to_slice::<u64>(
                combined_file
                    .get_posting_list(1)
                    .expect("Failed to read posting_list")
            ),
            posting_lists[1]
        );
        assert!(combined_file.get_posting_list(2).is_err());
    }
}
