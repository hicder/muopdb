use std::sync::Arc;

use anyhow::{anyhow, Result};
use byteorder::{ByteOrder, LittleEndian};
use utils::block_cache::BlockCache;

use crate::hnsw::writer::Header;

pub struct GraphOffsets {
    pub data_offset: usize,
    pub edges_offset: usize,
    pub points_offset: usize,
    pub edge_offsets_offset: usize,
    pub level_offsets_offset: usize,
    pub doc_id_mapping_offset: usize,
}

pub struct BlockBasedHnswGraphStorage {
    block_cache: Arc<BlockCache>,
    graph_file_id: u64,
    header: Header,
    offsets: GraphOffsets,
    level_offsets: Vec<u64>,
}

impl BlockBasedHnswGraphStorage {
    pub async fn new(block_cache: Arc<BlockCache>, base_directory: String) -> Result<Self> {
        let graph_path = format!("{}/hnsw/index", base_directory);
        let file_id = {
            let cache = block_cache.clone();
            cache.open_file(&graph_path).await
        }
        .map_err(|e| anyhow!("Failed to open graph file: {}", e))?;

        let header_data = {
            let cache = block_cache.clone();
            cache.read(file_id, 0, 49).await
        }
        .map_err(|e| anyhow!("Failed to read header: {}", e))?;

        let header = Self::parse_header(&header_data);
        let offsets = Self::calculate_offsets(&header, 49);

        let mut storage = Self {
            block_cache,
            graph_file_id: file_id,
            header,
            offsets,
            level_offsets: vec![],
        };
        storage.level_offsets = storage.get_level_offsets_slice().await?;

        Ok(storage)
    }

    pub async fn new_with_offset(
        block_cache: Arc<BlockCache>,
        base_directory: String,
        data_offset: usize,
    ) -> Result<Self> {
        let graph_path = format!("{}/hnsw/index", base_directory);
        let file_id = {
            let cache = block_cache.clone();
            cache.open_file(&graph_path).await
        }
        .map_err(|e| anyhow!("Failed to open graph file: {}", e))?;

        let header_data = {
            let cache = block_cache.clone();
            cache.read(file_id, data_offset as u64, 49).await
        }
        .map_err(|e| anyhow!("Failed to read header: {}", e))?;

        let header = Self::parse_header(&header_data);
        let mut offsets = Self::calculate_offsets(&header, data_offset + 49);
        // The data_offset in GraphOffsets is expected to be the start of the data block
        offsets.data_offset = data_offset + 49;

        let mut storage = Self {
            block_cache,
            graph_file_id: file_id,
            header,
            offsets,
            level_offsets: vec![],
        };
        storage.level_offsets = storage.get_level_offsets_slice().await?;

        Ok(storage)
    }

    fn parse_header(data: &[u8]) -> Header {
        let mut offset = 0;
        let version = match data[offset] {
            0 => crate::hnsw::writer::Version::V0,
            v => panic!("Unknown version: {}", v),
        };
        offset += 1;

        let quantized_dimension = LittleEndian::read_u32(&data[offset..]);
        offset += 4;

        let num_layers = LittleEndian::read_u32(&data[offset..]);
        offset += 4;

        let edges_len = LittleEndian::read_u64(&data[offset..]);
        offset += 8;

        let points_len = LittleEndian::read_u64(&data[offset..]);
        offset += 8;

        let edge_offsets_len = LittleEndian::read_u64(&data[offset..]);
        offset += 8;

        let level_offsets_len = LittleEndian::read_u64(&data[offset..]);
        offset += 8;

        let doc_id_mapping_len = LittleEndian::read_u64(&data[offset..]);

        Header {
            version,
            quantized_dimension,
            num_layers,
            edges_len,
            points_len,
            edge_offsets_len,
            level_offsets_len,
            doc_id_mapping_len,
        }
    }

    fn calculate_offsets(header: &Header, data_offset: usize) -> GraphOffsets {
        let offset = data_offset;
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

        GraphOffsets {
            data_offset,
            edges_offset,
            points_offset,
            edge_offsets_offset,
            level_offsets_offset,
            doc_id_mapping_offset,
        }
    }

    pub fn header(&self) -> &Header {
        &self.header
    }

    pub fn offsets(&self) -> &GraphOffsets {
        &self.offsets
    }

    /// NOTE: This method is very expensive as it loads the entire edges slice into memory.
    /// Use targeted reads if possible.
    pub async fn get_edges_slice(&self) -> Result<Vec<u32>> {
        let start = self.offsets.edges_offset as u64;
        let length = self.header.edges_len;
        if length == 0 {
            return Ok(vec![]);
        }
        let data = {
            let cache = self.block_cache.clone();
            cache.read(self.graph_file_id, start, length).await
        }
        .map_err(|e| anyhow!("Failed to read edges: {}", e))?;

        let num_elements = data.len() / 4;
        let mut result = Vec::with_capacity(num_elements);
        for i in 0..num_elements {
            let start = i * 4;
            let end = start + 4;
            let element = LittleEndian::read_u32(&data[start..end]);
            result.push(element);
        }
        Ok(result)
    }

    /// NOTE: This method is very expensive as it loads the entire edge offsets slice into memory.
    /// Use targeted reads if possible.
    pub async fn get_edge_offsets_slice(&self) -> Result<Vec<u64>> {
        let start = self.offsets.edge_offsets_offset as u64;
        let length = self.header.edge_offsets_len;
        if length == 0 {
            return Ok(vec![]);
        }
        let data = {
            let cache = self.block_cache.clone();
            cache.read(self.graph_file_id, start, length).await
        }
        .map_err(|e| anyhow!("Failed to read edge offsets: {}", e))?;

        let num_elements = data.len() / 8;
        let mut result = Vec::with_capacity(num_elements);
        for i in 0..num_elements {
            let start = i * 8;
            let end = start + 8;
            let element = LittleEndian::read_u64(&data[start..end]);
            result.push(element);
        }
        Ok(result)
    }

    /// NOTE: This method is very expensive as it loads the entire points slice into memory.
    /// Use targeted reads if possible.
    pub async fn get_points_slice(&self) -> Result<Vec<u32>> {
        let start = self.offsets.points_offset as u64;
        let length = self.header.points_len;
        if length == 0 {
            return Ok(vec![]);
        }
        let data = {
            let cache = self.block_cache.clone();
            cache.read(self.graph_file_id, start, length).await
        }
        .map_err(|e| anyhow!("Failed to read points: {}", e))?;

        let num_elements = data.len() / 4;
        let mut result = Vec::with_capacity(num_elements);
        for i in 0..num_elements {
            let start = i * 4;
            let end = start + 4;
            let element = LittleEndian::read_u32(&data[start..end]);
            result.push(element);
        }
        Ok(result)
    }

    /// Returns the level offsets slice.
    /// This is relatively small and already cached in `self.level_offsets`.
    pub async fn get_level_offsets_slice(&self) -> Result<Vec<u64>> {
        let start = self.offsets.level_offsets_offset as u64;
        let length = self.header.level_offsets_len;
        if length == 0 {
            return Ok(vec![]);
        }
        let data = {
            let cache = self.block_cache.clone();
            cache.read(self.graph_file_id, start, length).await
        }
        .map_err(|e| anyhow!("Failed to read level offsets: {}", e))?;

        let num_elements = data.len() / 8;
        let mut result = Vec::with_capacity(num_elements);
        for i in 0..num_elements {
            let start = i * 8;
            let end = start + 8;
            let element = LittleEndian::read_u64(&data[start..end]);
            result.push(element);
        }
        Ok(result)
    }

    /// NOTE: This method is very expensive as it loads the entire doc ID mapping slice into memory.
    /// Use targeted reads if possible.
    pub async fn get_doc_id_mapping_slice(&self) -> Result<Vec<u128>> {
        let start = self.offsets.doc_id_mapping_offset as u64;
        let length = self.header.doc_id_mapping_len;
        if length == 0 {
            return Ok(vec![]);
        }
        let data = {
            let cache = self.block_cache.clone();
            cache.read(self.graph_file_id, start, length).await
        }
        .map_err(|e| anyhow!("Failed to read doc ID mapping: {}", e))?;

        let num_elements = data.len() / 16;
        let mut result = Vec::with_capacity(num_elements);
        for i in 0..num_elements {
            let start = i * 16;
            let end = start + 16;
            let element = LittleEndian::read_u128(&data[start..end]);
            result.push(element);
        }
        Ok(result)
    }

    async fn get_u32_at(&self, offset: u64) -> Result<u32> {
        let data = self
            .block_cache
            .read(self.graph_file_id, offset, 4)
            .await
            .map_err(|e| anyhow!("Failed to read u32 at {}: {}", offset, e))?;
        Ok(LittleEndian::read_u32(&data))
    }

    async fn get_u64_at(&self, offset: u64) -> Result<u64> {
        let data = self
            .block_cache
            .read(self.graph_file_id, offset, 8)
            .await
            .map_err(|e| anyhow!("Failed to read u64 at {}: {}", offset, e))?;
        Ok(LittleEndian::read_u64(&data))
    }

    async fn get_u128_at(&self, offset: u64) -> Result<u128> {
        let data = self
            .block_cache
            .read(self.graph_file_id, offset, 16)
            .await
            .map_err(|e| anyhow!("Failed to read u128 at {}: {}", offset, e))?;
        Ok(LittleEndian::read_u128(&data))
    }

    async fn find_point_in_range(
        &self,
        point_id: u32,
        start_idx: usize,
        end_idx: usize,
    ) -> Result<Option<usize>> {
        const BATCH_SIZE: usize = 1024;
        let mut current = start_idx;
        let points_base = self.offsets.points_offset as u64;

        while current < end_idx {
            let batch_end = (current + BATCH_SIZE).min(end_idx);
            let read_len = (batch_end - current) * 4;
            let data = self
                .block_cache
                .read(
                    self.graph_file_id,
                    points_base + (current * 4) as u64,
                    read_len as u64,
                )
                .await?;

            for (i, chunk) in data.chunks_exact(4).enumerate() {
                if LittleEndian::read_u32(chunk) == point_id {
                    return Ok(Some(current + i));
                }
            }
            current = batch_end;
        }
        Ok(None)
    }

    pub async fn get_edges_for_point(&self, point_id: u32, layer: u8) -> Option<Vec<u32>> {
        let num_layers = self.header.num_layers as usize;
        if layer as usize >= num_layers {
            return None;
        }

        let level_idx_start = self.level_offsets[num_layers - 1 - layer as usize] as usize;
        let level_idx_end = self.level_offsets[num_layers - layer as usize] as usize;

        let idx_at_layer: i64;
        if layer > 0 {
            idx_at_layer = match self
                .find_point_in_range(point_id, level_idx_start, level_idx_end)
                .await
            {
                Ok(Some(idx)) => (idx - level_idx_start) as i64,
                _ => -1,
            };
        } else {
            idx_at_layer = point_id as i64;
        }

        if idx_at_layer < 0 {
            return None;
        }

        let idx = idx_at_layer as usize;
        let edge_offsets_base = self.offsets.edge_offsets_offset as u64;
        let start_idx_edges = match self
            .get_u64_at(edge_offsets_base + ((level_idx_start + idx) * 8) as u64)
            .await
        {
            Ok(v) => v,
            Err(_) => return None,
        };
        let end_idx_edges = match self
            .get_u64_at(edge_offsets_base + ((level_idx_start + idx + 1) * 8) as u64)
            .await
        {
            Ok(v) => v,
            Err(_) => return None,
        };

        if start_idx_edges == end_idx_edges {
            return None;
        }

        let edges_base = self.offsets.edges_offset as u64;
        let read_len = (end_idx_edges - start_idx_edges) * 4;
        let data = match self
            .block_cache
            .read(
                self.graph_file_id,
                edges_base + start_idx_edges * 4,
                read_len,
            )
            .await
        {
            Ok(v) => v,
            Err(_) => return None,
        };

        let mut result = Vec::with_capacity(data.len() / 4);
        for chunk in data.chunks_exact(4) {
            result.push(LittleEndian::read_u32(chunk));
        }
        Some(result)
    }

    pub async fn get_entry_point_top_layer(&self) -> u32 {
        if self.header.num_layers == 1 {
            let num_points = (self.header.edge_offsets_len / 8) as usize - 1;
            let edge_offsets_base = self.offsets.edge_offsets_offset as u64;
            let mut idx = 0;
            while idx < num_points {
                let edge_offset = self
                    .get_u64_at(edge_offsets_base + (idx * 8) as u64)
                    .await
                    .unwrap_or(0);
                let next_edge_offset = self
                    .get_u64_at(edge_offsets_base + ((idx + 1) * 8) as u64)
                    .await
                    .unwrap_or(0);
                if next_edge_offset > edge_offset {
                    return idx as u32;
                }
                idx += 1;
            }
            return 0;
        }

        let top_layer_start = self.level_offsets[0] as u64;
        let points_base = self.offsets.points_offset as u64;
        self.get_u32_at(points_base + top_layer_start * 4)
            .await
            .unwrap_or(0)
    }

    pub async fn map_point_id_to_doc_id(&self, point_ids: &[u32]) -> Result<Vec<u128>> {
        let mapping_base = self.offsets.doc_id_mapping_offset as u64;
        let mut result = Vec::with_capacity(point_ids.len());
        for &point_id in point_ids {
            let doc_id = self
                .get_u128_at(mapping_base + (point_id as u64 * 16))
                .await?;
            result.push(doc_id);
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use std::fs::{self, File};
    use std::io::Write;
    use std::sync::Arc;

    use tempdir::TempDir;
    use utils::block_cache::{BlockCache, BlockCacheConfig};

    use super::*;
    use crate::hnsw::writer::Version;

    fn create_test_file(
        base_directory: String,
        edges_len: u64,
        edge_offsets_len: u64,
        doc_id_mapping_len: u64,
    ) {
        let mut index_data: Vec<u8> = Vec::new();

        // Header
        index_data.push(0);
        index_data.extend_from_slice(&16_u32.to_le_bytes());
        index_data.extend_from_slice(&1_u32.to_le_bytes());
        index_data.extend_from_slice(&edges_len.to_le_bytes());
        index_data.extend_from_slice(&0_u64.to_le_bytes());
        index_data.extend_from_slice(&edge_offsets_len.to_le_bytes());
        index_data.extend_from_slice(&16_u64.to_le_bytes());
        index_data.extend_from_slice(&doc_id_mapping_len.to_le_bytes());

        // Edges padding
        let padding = (4 - (index_data.len() % 4)) % 4;
        index_data.extend(std::vec::from_elem(0u8, padding));

        // Edges
        index_data.extend_from_slice(&1_u32.to_le_bytes());
        index_data.extend_from_slice(&2_u32.to_le_bytes());
        index_data.extend_from_slice(&3_u32.to_le_bytes());

        // Edge offsets padding
        let padding = (8 - (index_data.len() % 8)) % 8;
        index_data.extend(std::vec::from_elem(0u8, padding));

        // Edge offsets (4 values)
        index_data.extend_from_slice(&0_u64.to_le_bytes());
        index_data.extend_from_slice(&1_u64.to_le_bytes());
        index_data.extend_from_slice(&1_u64.to_le_bytes());
        index_data.extend_from_slice(&2_u64.to_le_bytes());

        // Level offsets
        index_data.extend_from_slice(&0_u64.to_le_bytes());
        index_data.extend_from_slice(&0_u64.to_le_bytes());

        // Doc ID mapping
        index_data.extend_from_slice(&1_u128.to_le_bytes());
        index_data.extend_from_slice(&2_u128.to_le_bytes());

        let hnsw_dir = format!("{}/hnsw", base_directory);
        fs::create_dir_all(&hnsw_dir).unwrap();
        let mut index_file = File::create(format!("{}/index", hnsw_dir)).unwrap();
        index_file.write_all(&index_data).unwrap();
    }

    #[tokio::test]
    async fn test_async_graph_storage() {
        let temp_dir = TempDir::new("async_graph_storage_test").unwrap();
        let base_directory = temp_dir.path().to_str().unwrap().to_string();

        create_test_file(base_directory.clone(), 12, 32, 32);

        let config = BlockCacheConfig::default();
        let cache = Arc::new(BlockCache::new(config));

        let storage = BlockBasedHnswGraphStorage::new(cache.clone(), base_directory)
            .await
            .unwrap();

        assert_eq!(storage.header().version, Version::V0);
        assert_eq!(storage.header().num_layers, 1);

        let edges = storage.get_edges_slice().await.unwrap();
        assert_eq!(edges.len(), 3);
        assert_eq!(edges, vec![1, 2, 3]);

        let edge_offsets = storage.get_edge_offsets_slice().await.unwrap();
        assert_eq!(edge_offsets.len(), 4);

        let doc_ids = storage.get_doc_id_mapping_slice().await.unwrap();
        assert_eq!(doc_ids, vec![1, 2]);
    }
}
