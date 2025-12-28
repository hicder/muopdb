use std::collections::HashMap;
use std::mem::size_of;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use byteorder::{ByteOrder, LittleEndian};
use compression::compression::AsyncIntSeqDecoder;
use compression::elias_fano::block_based_decoder::BlockBasedEliasFanoDecoder;
use futures::future;
use utils::block_cache::cache::{BlockCache, FileId};
use utils::mem::transmute_u8_to_slice;

use crate::posting_list::combined_file::{Header, Version};

const PL_METADATA_LEN: usize = 2;
const MAX_READ_SIZE: u64 = 128;

/// Provides asynchronous access to IVF posting lists and related metadata stored on disk.
///
/// This storage handler manages the binary layout of the IVF index, including
/// document ID mappings, centroids, and compressed posting lists.
pub struct BlockBasedPostingListStorage {
    block_cache: Arc<BlockCache>,
    file_id: FileId,
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
    /// * `block_cache` - The shared block cache for file I/O.
    /// * `file_path` - The path to the IVF index file.
    ///
    /// # Returns
    /// * `Result<Self>` - A new storage handler instance or an error if initialization fails.
    pub async fn new(block_cache: Arc<BlockCache>, file_path: String) -> Result<Self> {
        Self::new_with_offset(block_cache, file_path, 0).await
    }

    /// Creates a new `AsyncPostingListStorage` handler starting at a specific file offset.
    ///
    /// # Arguments
    /// * `block_cache` - The shared block cache for file I/O.
    /// * `file_path` - The path to the IVF index file.
    /// * `offset` - The byte offset where the IVF storage data begins.
    ///
    /// # Returns
    /// * `Result<Self>` - A new storage handler instance or an error if initialization fails.
    pub async fn new_with_offset(
        block_cache: Arc<BlockCache>,
        file_path: String,
        offset: usize,
    ) -> Result<Self> {
        let file_id = block_cache
            .open_file(&file_path)
            .await
            .map_err(|e| anyhow!("Failed to open index file: {}", e))?;

        let header_data = block_cache
            .read(file_id, offset as u64, 64) // Read enough for header
            .await?;

        let (header, section_offset) = Self::read_header(&header_data, 0)?;
        let doc_id_mapping_offset = offset + section_offset;

        let centroid_offset = Self::align_to_next_boundary(
            doc_id_mapping_offset + header.doc_id_mapping_len as usize,
            8,
        );

        let metadata_section_offset =
            Self::align_to_next_boundary(centroid_offset + header.centroids_len as usize, 8);

        let count_data = block_cache
            .read(file_id, metadata_section_offset as u64, 8)
            .await?;
        let num_posting_lists = LittleEndian::read_u64(&count_data) as usize;

        let posting_list_metadata_offset = metadata_section_offset + size_of::<u64>();
        let posting_list_start_offset =
            posting_list_metadata_offset + num_posting_lists * PL_METADATA_LEN * size_of::<u64>();

        Ok(Self {
            block_cache,
            file_id,
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

        curr_offset = Self::align_to_next_boundary(curr_offset, 8);

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

        let data = self
            .block_cache
            .read(self.file_id, start as u64, 16)
            .await?;
        Ok(u128::from_le_bytes(
            data.try_into()
                .map_err(|_| anyhow!("Failed to read doc_id"))?,
        ))
    }

    /// Retrieves a batch of 128-bit document IDs for the given internal vector indices.
    ///
    /// Optimizes reads by grouping indices that share the same block and fall within
    /// the same 128-byte aligned region. This reduces the number of read requests
    /// to the block cache significantly.
    ///
    /// # Arguments
    /// * `indices` - A slice of internal indices of the vectors.
    ///
    /// # Returns
    /// * `Vec<Result<u128>>` - A vector of results, one per index.
    pub async fn get_doc_ids(&self, indices: &[usize]) -> Vec<Result<u128>> {
        if indices.is_empty() {
            return Vec::new();
        }

        let num_indices = indices.len();
        let mut results: Vec<Result<u128>> = Vec::with_capacity(num_indices);
        results.resize_with(num_indices, || Err(anyhow!("Not initialized")));

        let index_to_offset: Vec<(usize, Option<usize>)> = indices
            .iter()
            .enumerate()
            .map(|(result_idx, &index)| {
                if index >= self.header.num_vectors as usize {
                    return (result_idx, None);
                }
                let offset =
                    self.doc_id_mapping_offset + size_of::<u128>() + index * size_of::<u128>();
                (result_idx, Some(offset))
            })
            .collect();

        for (result_idx, offset_opt) in &index_to_offset {
            if offset_opt.is_none() {
                results[*result_idx] = Err(anyhow!("Index out of bound"));
            }
        }

        let index_to_offset: Vec<(usize, usize)> = index_to_offset
            .into_iter()
            .filter_map(|(idx, o)| o.map(|off| (idx, off)))
            .collect();

        if index_to_offset.is_empty() {
            return results;
        }

        let mut block_groups: HashMap<u64, Vec<(usize, usize)>> = HashMap::new();
        let block_size = self.block_cache.block_size();
        for (result_idx, offset) in index_to_offset.iter().copied() {
            let block = offset as u64 / block_size;
            block_groups
                .entry(block)
                .or_default()
                .push((result_idx, offset));
        }

        let mut read_futures = Vec::new();
        for (_block, offsets) in block_groups {
            let mut offsets = offsets.clone();
            offsets.sort_by_key(|&(_, o)| o);

            let mut i = 0;
            while i < offsets.len() {
                let start_offset = offsets[i].1;
                let chunk_index = start_offset / MAX_READ_SIZE as usize;
                let mut cluster_indices = Vec::new();

                while i < offsets.len() && offsets[i].1 / MAX_READ_SIZE as usize == chunk_index {
                    cluster_indices.push(offsets[i]);
                    i += 1;
                }

                let read_start = (chunk_index * MAX_READ_SIZE as usize) as u64;
                let read_length = MAX_READ_SIZE;
                let read_indices = cluster_indices;

                let file_id = self.file_id;
                let block_cache = self.block_cache.clone();
                read_futures.push(async move {
                    let data = block_cache.read(file_id, read_start, read_length).await?;
                    Ok((data, read_indices))
                });
            }
        }

        let read_results: Vec<Result<(Vec<u8>, Vec<(usize, usize)>), anyhow::Error>> =
            future::join_all(read_futures).await;

        let mut read_data_map: HashMap<(u64, usize), Vec<u8>> = HashMap::new();
        let outer_indices = indices;
        for read_result in read_results {
            match read_result {
                Ok((data, indices)) => {
                    for (_result_idx, offset) in indices {
                        let key = (offset as u64 / block_size, offset / MAX_READ_SIZE as usize);
                        if !read_data_map.contains_key(&key) {
                            read_data_map.insert(key, data.clone());
                        }
                    }
                }
                Err(e) => {
                    for &result_idx in outer_indices.iter() {
                        if results[result_idx].is_err() {
                            results[result_idx] = Err(anyhow!("{}", e));
                        }
                    }
                }
            }
        }

        for (result_idx, offset) in index_to_offset {
            let key = (offset as u64 / block_size, offset / MAX_READ_SIZE as usize);
            match read_data_map.get(&key) {
                Some(data) => {
                    let pos = offset % MAX_READ_SIZE as usize;
                    let doc_id_bytes: [u8; 16] = match data[pos..pos + 16].try_into() {
                        Ok(bytes) => bytes,
                        Err(_) => {
                            results[result_idx] = Err(anyhow!("Failed to read doc_id"));
                            continue;
                        }
                    };
                    results[result_idx] = Ok(u128::from_le_bytes(doc_id_bytes));
                }
                None => {
                    results[result_idx] = Err(anyhow!("Failed to read doc_id"));
                }
            }
        }

        results
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

        let data = self
            .block_cache
            .read(self.file_id, start as u64, length as u64)
            .await?;
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

        let metadata_data = self
            .block_cache
            .read(self.file_id, metadata_offset as u64, 16)
            .await?;
        let _pl_len = LittleEndian::read_u64(&metadata_data[0..8]) as usize;
        let pl_offset =
            LittleEndian::read_u64(&metadata_data[8..16]) as usize + self.posting_list_start_offset;

        BlockBasedEliasFanoDecoder::new_decoder(
            self.block_cache.clone(),
            self.file_id,
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
