use memmap2::Mmap;
use quantization::{pq::ProductQuantizerReader, quantization::Quantizer};
use rand::Rng;
use std::{fs::File, vec};

use crate::hnsw::writer::Header;

use super::utils::{GraphTraversal, SearchContext};

pub struct Hnsw {
    // Need this for mmap
    #[allow(dead_code)]
    backing_file: File,
    mmap: Mmap,
    header: Header,
    data_offset: usize,
    quantizer: Box<dyn Quantizer>,

    // TODO(hicder): Populate this as well.
    // TODO(hicder): mmap this so we don't store all vectors in memory.
    vectors: Vec<Vec<u8>>,
}

impl Hnsw {
    pub fn new(
        backing_file: File,
        mmap: Mmap,
        header: Header,
        data_offset: usize,
        base_directory: String,
    ) -> Self {
        // Read quantizer
        let pq_reader = ProductQuantizerReader::new(base_directory.clone());
        let pq = pq_reader.read().unwrap();

        Self {
            backing_file,
            mmap,
            header,
            data_offset,
            quantizer: Box::new(pq),
            vectors: vec![],
        }
    }

    pub fn ann_search(&self, query: &[f32], k: usize, ef: u32) -> Vec<u32> {
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
        working_set.iter().map(|x| x.point_id).collect()
    }

    pub fn get_header(&self) -> &Header {
        &self.header
    }

    pub fn get_data_offset(&self) -> usize {
        self.data_offset
    }

    fn get_vector(&self, point_id: u32) -> &[u8] {
        &self.vectors[point_id as usize]
    }

    fn get_edges_slice(&self) -> &[u32] {
        let start = self.data_offset;
        utils::mem::transmute_u8_to_slice(&self.mmap[start..start + self.header.edges_len as usize])
    }

    fn get_points_slice(&self) -> &[u32] {
        let start = self.data_offset + self.header.edges_len as usize;
        utils::mem::transmute_u8_to_slice(
            &self.mmap[start..start + self.header.points_len as usize],
        )
    }

    /// Returns the edge offsets slice
    fn get_edge_offsets_slice(&self) -> &[u64] {
        let start =
            self.data_offset + self.header.edges_len as usize + self.header.points_len as usize;
        utils::mem::transmute_u8_to_slice(
            &self.mmap[start..start + self.header.edge_offsets_len as usize],
        )
    }

    /// Returns the level offsets slice
    fn get_level_offsets_slice(&self) -> &[u64] {
        let start = self.data_offset
            + self.header.edges_len as usize
            + self.header.points_len as usize
            + self.header.edge_offsets_len as usize;
        let slice = &self.mmap[start..start + self.header.level_offsets_len as usize];
        return utils::mem::transmute_u8_to_slice(slice);
    }

    fn get_entry_point_top_layer(&self) -> u32 {
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
}

impl GraphTraversal for Hnsw {
    fn distance(&self, query: &[u8], point_id: u32) -> f32 {
        self.quantizer.distance(query, self.get_vector(point_id))
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
            let _ = points.iter().enumerate().map(|(idx, point)| {
                if *point == point_id {
                    idx_at_layer = idx as i64;
                }
            });
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

        // This could only happen at layer 0.
        if start_idx_edges == end_idx_edges {
            assert!(layer == 0);
            return None;
        }

        let edges = &self.get_edges_slice()[start_idx_edges as usize..end_idx_edges as usize];
        Some(edges.to_vec())
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
