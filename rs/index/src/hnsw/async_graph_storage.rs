use std::sync::Arc;

use anyhow::{anyhow, Result};
use byteorder::{ByteOrder, LittleEndian};
use utils::block_cache::BlockCache;

use super::writer::Header;

pub struct GraphOffsets {
    pub data_offset: usize,
    pub edges_offset: usize,
    pub points_offset: usize,
    pub edge_offsets_offset: usize,
    pub level_offsets_offset: usize,
    pub doc_id_mapping_offset: usize,
}

pub struct AsyncHnswGraphStorage {
    block_cache: Arc<BlockCache>,
    graph_file_id: u64,
    header: Header,
    offsets: GraphOffsets,
}

impl AsyncHnswGraphStorage {
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

        Ok(Self {
            block_cache,
            graph_file_id: file_id,
            header,
            offsets,
        })
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

        Ok(Self {
            block_cache,
            graph_file_id: file_id,
            header,
            offsets,
        })
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

    pub async fn get_edges_for_point(&self, point_id: u32, layer: u8) -> Option<Vec<u32>> {
        let num_layers = self.header.num_layers as usize;
        let level_offsets = match self.get_level_offsets_slice().await {
            Ok(v) => v,
            Err(_) => return None,
        };
        let edge_offsets = match self.get_edge_offsets_slice().await {
            Ok(v) => v,
            Err(_) => return None,
        };

        if layer as usize >= num_layers {
            return None;
        }

        let level_idx_start = level_offsets[num_layers - 1 - layer as usize] as usize;
        let level_idx_end = level_offsets[num_layers - layer as usize] as usize;

        let idx_at_layer: i64;
        if layer > 0 {
            let points = match self.get_points_slice().await {
                Ok(v) => v,
                Err(_) => return None,
            };
            idx_at_layer = points[level_idx_start..level_idx_end]
                .iter()
                .position(|&p| p == point_id)
                .map(|i| i as i64)
                .unwrap_or(-1);
        } else {
            idx_at_layer = point_id as i64;
        }

        if idx_at_layer < 0 {
            return None;
        }

        let idx = idx_at_layer as usize;
        let start_idx_edges = edge_offsets[level_idx_start + idx];
        let end_idx_edges = edge_offsets[level_idx_start + idx + 1];

        if start_idx_edges == end_idx_edges {
            return None;
        }

        let edges = match self.get_edges_slice().await {
            Ok(v) => v,
            Err(_) => return None,
        };

        Some(edges[start_idx_edges as usize..end_idx_edges as usize].to_vec())
    }

    pub async fn get_entry_point_top_layer(&self) -> u32 {
        if self.header.num_layers == 1 {
            let edge_offsets = match self.get_edge_offsets_slice().await {
                Ok(v) => v,
                Err(_) => return 0,
            };
            let mut idx = 0;
            while idx < edge_offsets.len() - 1 {
                let edge_offset = edge_offsets[idx];
                let next_edge_offset = edge_offsets[idx + 1];
                if next_edge_offset - edge_offset > 0 {
                    return idx as u32;
                }
                idx += 1;
            }
            return 0;
        }

        let level_offsets = match self.get_level_offsets_slice().await {
            Ok(v) => v,
            Err(_) => return 0,
        };
        let points = match self.get_points_slice().await {
            Ok(v) => v,
            Err(_) => return 0,
        };
        let top_layer_points = &points[level_offsets[0] as usize..level_offsets[1] as usize];
        if top_layer_points.is_empty() {
            return 0;
        }
        top_layer_points[0]
    }

    pub async fn map_point_id_to_doc_id(&self, point_ids: &[u32]) -> Result<Vec<u128>> {
        let doc_id_mapping = self.get_doc_id_mapping_slice().await?;
        Ok(point_ids
            .iter()
            .map(|x| doc_id_mapping[*x as usize])
            .collect())
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

        let storage = AsyncHnswGraphStorage::new(cache.clone(), base_directory)
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
