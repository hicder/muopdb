use std::fs::File;

use memmap2::Mmap;
use quantization::pq::ProductQuantizerReader;
use quantization::quantization::Quantizer;
use rand::Rng;
use roaring::RoaringBitmap;

use super::utils::{GraphTraversal, TraversalContext};
use crate::hnsw::writer::Header;
use crate::index::Index;

pub struct SearchContext {
    visited: RoaringBitmap,
}

impl SearchContext {
    pub fn new() -> Self {
        Self {
            visited: RoaringBitmap::new(),
        }
    }
}

impl TraversalContext for SearchContext {
    fn visited(&self, i: u32) -> bool {
        self.visited.contains(i)
    }

    fn set_visited(&mut self, i: u32) {
        self.visited.insert(i);
    }
}

pub struct Hnsw {
    // Need this for mmap
    #[allow(dead_code)]
    backing_file: File,
    #[allow(dead_code)]
    vector_storage_file: File,
    mmap: Mmap,
    vector_storage_mmap: Mmap,
    header: Header,
    data_offset: usize,
    edges_offset: usize,
    points_offset: usize,
    edge_offsets_offset: usize,
    level_offsets_offset: usize,
    doc_id_mapping_offset: usize,

    quantizer: Box<dyn Quantizer + Send + Sync>,
}

impl Hnsw {
    pub fn new(
        backing_file: File,
        vector_storage_file: File,
        header: Header,
        data_offset: usize,
        edges_offset: usize,
        points_offset: usize,
        edge_offsets_offset: usize,
        level_offsets_offset: usize,
        doc_id_mapping_offset: usize,
        base_directory: String,
    ) -> Self {
        // Read quantizer
        let pq_directory = format!("{}/quantizer", base_directory);
        let pq_reader = ProductQuantizerReader::new(pq_directory);
        let pq = pq_reader.read().unwrap();

        let index_mmap = unsafe { Mmap::map(&backing_file).unwrap() };
        let vector_storage_mmap = unsafe { Mmap::map(&vector_storage_file).unwrap() };

        Self {
            backing_file,
            vector_storage_file,
            mmap: index_mmap,
            vector_storage_mmap,
            header,
            data_offset,
            edges_offset,
            points_offset,
            edge_offsets_offset,
            level_offsets_offset,
            doc_id_mapping_offset,
            quantizer: Box::new(pq),
        }
    }

    fn map_point_id_to_doc_id(&self, point_ids: &[u32]) -> Vec<u64> {
        let doc_id_mapping = self.get_doc_id_mapping_slice();
        point_ids
            .iter()
            .map(|x| doc_id_mapping[*x as usize])
            .collect()
    }

    pub fn ann_search(&self, query: &[f32], k: usize, ef: u32) -> Vec<u64> {
        let mut context = SearchContext::new();

        let quantized_query = self.quantizer.quantize(query);
        let mut current_layer: i32 = self.header.num_layers as i32 - 1;
        let mut ep = self.get_entry_point_top_layer();
        let mut working_set;
        while current_layer > 0 {
            working_set =
                self.search_layer(&mut context, &quantized_query, ep, 1, current_layer as u8);
            ep = working_set
                .iter()
                .min_by(|x, y| x.distance.cmp(&y.distance))
                .unwrap()
                .point_id;
            current_layer -= 1;
        }

        working_set = self.search_layer(&mut context, &quantized_query, ep, ef, 0);
        working_set.sort_by(|x, y| x.distance.cmp(&y.distance));
        working_set.truncate(k);
        let point_ids: Vec<u32> = working_set.iter().map(|x| x.point_id).collect();
        self.map_point_id_to_doc_id(&point_ids)
    }

    pub fn get_header(&self) -> &Header {
        &self.header
    }

    pub fn get_data_offset(&self) -> usize {
        self.data_offset
    }

    fn get_vector(&self, point_id: u32) -> &[u8] {
        &self.get_vector_storage_slice()[point_id as usize
            * self.header.quantized_dimension as usize
            ..(point_id + 1) as usize * self.header.quantized_dimension as usize]
    }

    fn get_vector_storage_slice(&self) -> &[u8] {
        utils::mem::transmute_u8_to_slice(&self.vector_storage_mmap)
    }

    pub fn get_edges_slice(&self) -> &[u32] {
        let start = self.edges_offset;
        utils::mem::transmute_u8_to_slice(&self.mmap[start..start + self.header.edges_len as usize])
    }

    pub fn get_points_slice(&self) -> &[u32] {
        let start = self.points_offset;
        utils::mem::transmute_u8_to_slice(
            &self.mmap[start..start + self.header.points_len as usize],
        )
    }

    /// Returns the edge offsets slice
    pub fn get_edge_offsets_slice(&self) -> &[u64] {
        let start = self.edge_offsets_offset;
        utils::mem::transmute_u8_to_slice(
            &self.mmap[start..start + self.header.edge_offsets_len as usize],
        )
    }

    /// Returns the level offsets slice
    pub fn get_level_offsets_slice(&self) -> &[u64] {
        let start = self.level_offsets_offset;
        let slice =
            &self.mmap[self.level_offsets_offset..start + self.header.level_offsets_len as usize];
        return utils::mem::transmute_u8_to_slice(slice);
    }

    /// Returns the doc_id_mapping slice
    pub fn get_doc_id_mapping_slice(&self) -> &[u64] {
        let start = self.doc_id_mapping_offset;
        let slice =
            &self.mmap[self.doc_id_mapping_offset..start + self.header.doc_id_mapping_len as usize];
        return utils::mem::transmute_u8_to_slice(slice);
    }

    pub fn get_entry_point_top_layer(&self) -> u32 {
        // If we only have bottom layer, just return a random one.
        if self.header.num_layers == 1 {
            let mut idx = 0;
            let edge_offsets_slice = self.get_edge_offsets_slice();
            while idx < edge_offsets_slice.len() - 1 {
                let edge_offset = edge_offsets_slice[idx];
                let next_edge_offset = edge_offsets_slice[idx + 1];
                if next_edge_offset - edge_offset > 0 {
                    return idx as u32;
                }
                idx += 1;
            }

            // Should be unreachable here.
            return 0;
        }

        // Just pick a random point at the top layer.
        let level_offsets_slice = self.get_level_offsets_slice();
        let points = &self.get_points_slice()
            [level_offsets_slice[0] as usize..level_offsets_slice[1] as usize];
        let mut rng = rand::thread_rng();
        points[rng.gen_range(0..points.len())]
    }

    #[cfg(test)]
    pub fn get_doc_id_test(&self, point_ids: &[u32]) -> Vec<u64> {
        let doc_id_mapping = self.get_doc_id_mapping_slice();
        point_ids
            .iter()
            .map(|x| doc_id_mapping[*x as usize])
            .collect()
    }
}

impl GraphTraversal for Hnsw {
    type ContextT = SearchContext;

    fn distance(&self, query: &[u8], point_id: u32) -> f32 {
        self.quantizer.distance(
            query,
            self.get_vector(point_id),
            utils::l2::L2DistanceCalculatorImpl::StreamingWithSIMD,
        )
    }

    fn get_edges_for_point(&self, point_id: u32, layer: u8) -> Option<Vec<u32>> {
        let num_layers = self.header.num_layers as usize;
        let level_idx_start =
            self.get_level_offsets_slice()[num_layers - 1 - layer as usize] as usize;
        let level_idx_end = self.get_level_offsets_slice()[num_layers - layer as usize] as usize;

        // id of into edge_offsets at current layer.
        // note that this starts at 0, so we need to add level_idx_start to get the actual idx
        let mut idx_at_layer = -1 as i64;

        if layer > 0 {
            let points = &self.get_points_slice()[level_idx_start..level_idx_end];
            for i in 0..points.len() {
                if points[i] == point_id {
                    idx_at_layer = i as i64;
                    break;
                }
            }
        } else {
            // At layer 0, we have all points.
            // TODO(hicder): Check that point_id is within range.
            idx_at_layer = point_id as i64;
        }

        if idx_at_layer < 0 {
            return None;
        }

        let idx = idx_at_layer as usize;
        let start_idx_edges = self.get_edge_offsets_slice()[level_idx_start + idx];
        let end_idx_edges = self.get_edge_offsets_slice()[level_idx_start + idx + 1];

        if start_idx_edges == end_idx_edges {
            return None;
        }

        let edges = &self.get_edges_slice()[start_idx_edges as usize..end_idx_edges as usize];
        Some(edges.to_vec())
    }

    fn print_graph(&self, layer: u8, predicate: impl Fn(u8, u32) -> bool) {
        let num_layers = self.header.num_layers as usize;
        if layer as usize >= num_layers {
            println!("Layer {} is out of range.", layer);
            return;
        }

        let level_idx_start =
            self.get_level_offsets_slice()[num_layers - 1 - layer as usize] as usize;
        let level_idx_end = self.get_level_offsets_slice()[num_layers - layer as usize] as usize;

        if layer > 0 {
            let points = &self.get_points_slice()[level_idx_start..level_idx_end];
            println!("Layer {}, number of points: {}", layer, points.len());
            for i in 0..points.len() {
                if !predicate(layer, i as u32) {
                    continue;
                }
                let idx = i as usize;

                let start_idx_edges = self.get_edge_offsets_slice()[level_idx_start + idx];
                let end_idx_edges = self.get_edge_offsets_slice()[level_idx_start + idx + 1];

                print!("{} -> ", points[idx]);

                if start_idx_edges == end_idx_edges {
                    return;
                }

                let edges =
                    &self.get_edges_slice()[start_idx_edges as usize..end_idx_edges as usize];
                for e in edges {
                    print!("{}, ", e);
                }
                println!("");
            }
        } else {
            let num_points = level_idx_end - level_idx_start;
            println!("Layer {}, number of points: {}", layer, num_points);
            for i in 0..num_points {
                if !predicate(layer, i as u32) {
                    continue;
                }

                let idx = i as usize;
                let start_idx_edges = self.get_edge_offsets_slice()[level_idx_start + idx];
                let end_idx_edges = self.get_edge_offsets_slice()[level_idx_start + idx + 1];

                print!("{} -> ", idx);

                if start_idx_edges == end_idx_edges {
                    return;
                }

                let edges =
                    &self.get_edges_slice()[start_idx_edges as usize..end_idx_edges as usize];
                for e in edges {
                    print!("{}, ", e);
                }
                println!("");
            }
        }
    }
}

impl Index for Hnsw {
    fn search(&self, query: &[f32], k: usize) -> Option<Vec<u64>> {
        // TODO(hicder): Add ef parameter
        Some(self.ann_search(query, k, 10))
    }
}

// Test
#[cfg(test)]
mod tests {
    use std::io::Read;

    #[test]
    fn test_hnsw() {
        println!("{}", env!("CARGO_MANIFEST_DIR"));
        let dataset_file = std::fs::File::open(format!(
            "{}/resources/10000_rows_128_dim",
            env!("CARGO_MANIFEST_DIR")
        ));

        let mut buffer_reader = std::io::BufReader::new(dataset_file.unwrap());
        let mut buffer: [u8; 4] = [0; 4];
        let mut dataset: Vec<Vec<f32>> = vec![];
        for _ in 0..10000 {
            let mut v = Vec::<f32>::with_capacity(128);
            for _i in 0..128 {
                buffer_reader.read(&mut buffer).unwrap();
                v.push(f32::from_le_bytes(buffer));
            }
            dataset.push(v);
        }

        assert_eq!(dataset.len(), 10000);
    }
}
