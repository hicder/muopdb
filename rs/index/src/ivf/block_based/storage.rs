use std::mem::size_of;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use byteorder::{ByteOrder, LittleEndian};
use compression::compression::AsyncIntSeqDecoder;
use compression::elias_fano::block_based_decoder::BlockBasedEliasFanoDecoder;
use utils::file_io::env::{Env, OpenResult};
use utils::file_io::FileIO;
use utils::mem::transmute_u8_to_slice;

use crate::posting_list::combined_file::{Header, Version};

const PL_METADATA_LEN: usize = 2;

/// Provides asynchronous access to IVF posting lists and related metadata stored on disk.
///
/// This storage handler manages the binary layout of the IVF index, including
/// document ID mappings, centroids, and compressed posting lists.
pub struct BlockBasedPostingListStorage {
    file_io: Arc<dyn FileIO + Send + Sync>,
    header: Header,
    doc_id_mapping_offset: usize,
    centroid_offset: usize,
    posting_list_metadata_offset: usize,
    posting_list_start_offset: usize,
    num_posting_lists: usize,
}

impl BlockBasedPostingListStorage {
    /// Creates a new `AsyncPostingListStorage` handler for the specified file.
    ///
    /// # Arguments
    /// * `env` - The environment for file I/O.
    /// * `file_path` - The path to the IVF index file.
    ///
    /// # Returns
    /// * `Result<Self>` - A new storage handler instance or an error if initialization fails.
    pub async fn new(env: Arc<Box<dyn Env>>, file_path: String) -> Result<Self> {
        Self::new_with_offset(env, file_path, 0).await
    }

    /// Creates a new `AsyncPostingListStorage` handler starting at a specific file offset.
    ///
    /// # Arguments
    /// * `env` - The environment for file I/O.
    /// * `file_path` - The path to the IVF index file.
    /// * `offset` - The byte offset where the IVF storage data begins.
    ///
    /// # Returns
    /// * `Result<Self>` - A new storage handler instance or an error if initialization fails.
    pub async fn new_with_offset(env: Arc<Box<dyn Env>>, file_path: String, offset: usize) -> Result<Self> {
        let OpenResult { file_io, .. } = env.open(&file_path)
            .await
            .map_err(|e| anyhow!("Failed to open index file: {}", e))?;

        let header_data = file_io.read(offset as u64, 64).await?;

        let (header, section_offset) = Self::read_header(&header_data, 0)?;
        let doc_id_mapping_offset = offset + section_offset;

        let centroid_offset = Self::align_to_next_boundary(
            doc_id_mapping_offset + header.doc_id_mapping_len as usize,
            8,
        );

        let metadata_section_offset =
            Self::align_to_next_boundary(centroid_offset + header.centroids_len as usize, 8);

        let count_data = file_io.read(metadata_section_offset as u64, 8).await?;
        let num_posting_lists = LittleEndian::read_u64(&count_data) as usize;

        let posting_list_metadata_offset = metadata_section_offset + size_of::<u64>();
        let posting_list_start_offset =
            posting_list_metadata_offset + num_posting_lists * PL_METADATA_LEN * size_of::<u64>();

        Ok(Self {
            file_io,
            header,
            doc_id_mapping_offset,
            centroid_offset,
            posting_list_metadata_offset,
            posting_list_start_offset,
            num_posting_lists,
        })
    }

    /// Parses the IVF storage header from a byte buffer.
    ///
    /// # Arguments
    /// * `buffer` - The byte buffer containing the encoded header.
    /// * `offset` - The starting position within the buffer.
    ///
    /// # Returns
    /// * `Result<(Header, usize)>` - The parsed header and the size of the header section.
    fn read_header(buffer: &[u8], offset: usize) -> Result<(Header, usize)> {
        let mut curr_offset = offset;
        let version = match buffer[curr_offset] {
            0 => Version::V0,
            default => return Err(anyhow!("Unknown version: {}", default)),
        };
        curr_offset += 1;

        let num_features = LittleEndian::read_u32(&buffer[curr_offset..]);
        curr_offset += 4;
        let quantized_dimension = LittleEndian::read_u32(&buffer[curr_offset..]);
        curr_offset += 4;
        let num_clusters = LittleEndian::read_u32(&buffer[curr_offset..]);
        curr_offset += 4;
        let num_vectors = LittleEndian::read_u64(&buffer[curr_offset..]);
        curr_offset += 8;
        let doc_id_mapping_len = LittleEndian::read_u64(&buffer[curr_offset..]);
        curr_offset += 8;
        let centroids_len = LittleEndian::read_u64(&buffer[curr_offset..]);
        curr_offset += 8;
        let posting_lists_and_metadata_len = LittleEndian::read_u64(&buffer[curr_offset..]);
        curr_offset += 8;

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

        curr_offset = Self::align_to_next_boundary(curr_offset, 16);

        Ok((header, curr_offset))
    }

    /// Aligns a position to the next multiple of the specified alignment.
    ///
    /// # Arguments
    /// * `current_position` - The current byte position.
    /// * `alignment` - The desired byte alignment (must be a power of two).
    ///
    /// # Returns
    /// * `usize` - The aligned byte position.
    fn align_to_next_boundary(current_position: usize, alignment: usize) -> usize {
        let mask = alignment - 1;
        (current_position + mask) & !mask
    }

    /// Retrieves a 128-bit document ID for a given internal vector index.
    ///
    /// # Arguments
    /// * `index` - The internal index of the vector.
    ///
    /// # Returns
    /// * `Result<u128>` - The 128-bit document ID or an error if the index is out of bounds.
    pub async fn get_doc_id(&self, index: usize) -> Result<u128> {
        if index >= self.header.num_vectors as usize {
            return Err(anyhow!("Index out of bound"));
        }

        let start = self.doc_id_mapping_offset + size_of::<u128>() + index * size_of::<u128>();

        let data = self.file_io.read(start as u64, 16).await?;
        Ok(u128::from_le_bytes(
            data.try_into()
                .map_err(|_| anyhow!("Failed to read doc_id"))?,
        ))
    }

    /// Retrieves the centroid vector for a given cluster index.
    ///
    /// # Arguments
    /// * `index` - The index of the cluster.
    ///
    /// # Returns
    /// * `Result<Vec<f32>>` - The centroid vector or an error if the index is out of bounds.
    pub async fn get_centroid(&self, index: usize) -> Result<Vec<f32>> {
        if index >= self.header.num_clusters as usize {
            return Err(anyhow!("Index out of bound"));
        }

        let start = self.centroid_offset
            + size_of::<u64>()
            + index * self.header.num_features as usize * size_of::<f32>();
        let length = self.header.num_features as usize * size_of::<f32>();

        let data = self.file_io.read(start as u64, length as u64).await?;
        Ok(transmute_u8_to_slice::<f32>(&data).to_vec())
    }

    /// Creates an asynchronous decoder for a specific posting list.
    ///
    /// # Arguments
    /// * `index` - The index of the cluster/posting list to decode.
    ///
    /// # Returns
    /// * `Result<BlockBasedEliasFanoDecoder<u64>>` - A decoder for the posting list.
    pub async fn get_posting_list_decoder(
        &self,
        index: usize,
    ) -> Result<BlockBasedEliasFanoDecoder<u64>> {
        if index >= self.num_posting_lists {
            return Err(anyhow!(
                "Index {} out of bound (num_posting_lists: {})",
                index,
                self.num_posting_lists
            ));
        }

        let metadata_offset =
            self.posting_list_metadata_offset + index * PL_METADATA_LEN * size_of::<u64>();

        let metadata_data = self.file_io.read(metadata_offset as u64, 16).await?;
        let _pl_len = LittleEndian::read_u64(&metadata_data[0..8]) as usize;
        let pl_offset =
            LittleEndian::read_u64(&metadata_data[8..16]) as usize + self.posting_list_start_offset;

        BlockBasedEliasFanoDecoder::new_decoder(
            self.file_io.clone(),
            pl_offset as u64,
            4096, // default buffer size
        )
        .await
    }

    /// Returns a reference to the IVF storage header.
    ///
    /// # Returns
    /// * `&Header` - A reference to the internal header structure.
    pub fn header(&self) -> &Header {
        &self.header
    }
}
