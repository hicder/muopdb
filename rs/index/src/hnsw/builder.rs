use std::cmp::min;
use std::collections::{BinaryHeap, HashMap, HashSet};

use ordered_float::NotNan;
use quantization::quantization::Quantizer;
use rand::Rng;

use super::utils::{GraphTraversal, PointAndDistance, SearchContext};

/// TODO(hicder): support bare vector in addition to quantized one.
pub struct Layer {
    pub edges: HashMap<u32, Vec<PointAndDistance>>,
}

/// The actual builder
pub struct HnswBuilder {
    vectors: Vec<Vec<u8>>,

    max_neighbors: usize,
    pub layers: Vec<Layer>,
    pub current_top_layer: u8,
    quantizer: Box<dyn Quantizer>,
    ef_contruction: u32,
    pub entry_point: Vec<u32>,
    max_layer: u8,
}

// TODO(hicder): support bare vector in addition to quantized one.
// TODO(hicder): Reindex to make all connected points have close ids so that we fetch together.
impl HnswBuilder {
    pub fn new(
        max_neighbors: usize,
        max_layers: u8,
        ef_construction: u32,
        quantizer: Box<dyn Quantizer>,
    ) -> Self {
        Self {
            vectors: vec![],
            max_neighbors,
            max_layer: max_layers,
            layers: vec![],
            current_top_layer: 0,
            quantizer: quantizer,
            ef_contruction: ef_construction,
            entry_point: vec![],
        }
    }

    pub fn save_vector(&mut self, vector: &[u8], point_id: u32) {
        // Check if vectors are large enough. If not extend the vectors
        if self.vectors.len() < point_id as usize + 1 {
            self.vectors.resize(point_id as usize + 1, vec![]);
        }
        self.vectors[point_id as usize] = vector.to_vec();
    }

    /// Insert a vector into the index
    pub fn insert(&mut self, doc_id: u64, vector: &[f32]) {
        let mut context = SearchContext::new();
        let quantized_query = self.quantizer.quantize(vector);
        // TODO(hicder): Use id provider instead of doc_id
        let point_id = doc_id as u32;

        let empty_graph = self.vectors.is_empty();
        // Save the vector
        self.save_vector(&quantized_query, point_id);
        let layer = self.get_random_layer();

        if empty_graph {
            self.entry_point = vec![point_id];
            // Initialize the layers
            for _ in 0..=layer {
                self.layers.push(Layer {
                    edges: HashMap::from([(point_id, vec![])]),
                });
            }
            self.current_top_layer = layer;
            return;
        }

        let mut entry_point = self.entry_point[0];
        if layer < self.current_top_layer {
            for l in ((layer + 1)..self.current_top_layer).rev() {
                let nearest_elements =
                    self.search_layer(&mut context, &quantized_query, entry_point, 1, l);
                entry_point = nearest_elements[0].point_id;
            }
        } else if layer > self.current_top_layer {
            // Initialize the layers
            for _ in 0..(layer - self.current_top_layer) {
                self.layers.push(Layer {
                    edges: HashMap::from([(point_id, vec![])]),
                });
            }
        }

        for l in (0..=min(layer, self.current_top_layer)).rev() {
            let nearest_elements = self.search_layer(
                &mut context,
                &quantized_query,
                entry_point,
                self.ef_contruction,
                l,
            );
            let neighbors =
                self.select_neighbors_heuristic(&nearest_elements, self.max_neighbors as usize);
            for e in &neighbors {
                self.layers[l as usize]
                    .edges
                    .entry(e.point_id)
                    .or_insert_with(|| vec![])
                    .push(PointAndDistance {
                        point_id,
                        distance: e.distance.clone(),
                    });
                self.layers[l as usize]
                    .edges
                    .entry(point_id)
                    .or_insert_with(|| vec![])
                    .push(e.clone());
            }

            for e in &neighbors {
                let e_edges = &self.layers[l as usize].edges[&e.point_id];
                let num_edges = e_edges.len();
                if num_edges > self.max_neighbors as usize {
                    // Trim the edges
                    let new_edges_for_e =
                        self.select_neighbors_heuristic(&e_edges, self.max_neighbors);
                    self.layers[l as usize]
                        .edges
                        .insert(e.point_id, new_edges_for_e);
                }
            }
            entry_point = nearest_elements[0].point_id;
        }

        if layer > self.current_top_layer {
            self.current_top_layer = layer;
            self.entry_point = vec![point_id];
        } else if layer == self.current_top_layer {
            self.entry_point.push(point_id);
        }
    }

    pub fn get_nodes_from_non_bottom_layer(&self) -> Vec<u32> {
        let mut nodes = vec![];
        let mut current_layer = self.current_top_layer;
        let mut visited: HashSet<u32> = HashSet::new();
        while current_layer > 0 {
            for (point_id, _) in self.layers[current_layer as usize].edges.iter() {
                if !visited.contains(point_id) {
                    nodes.push(*point_id);
                    visited.insert(*point_id);
                }
            }
            current_layer -= 1;
        }
        nodes
    }

    /// Compute the distance between two points.
    /// The vectors are quantized.
    fn distance_two_points(&self, a: u32, b: u32) -> f32 {
        let a_vector = self.get_vector(a);
        let b_vector = self.get_vector(b);
        self.quantizer.distance(a_vector, b_vector)
    }

    fn get_vector(&self, point_id: u32) -> &[u8] {
        &self.vectors[point_id as usize]
    }

    fn get_random_layer(&self) -> u8 {
        let mut rng = rand::thread_rng();
        let random = rng.gen::<f32>();
        ((-random.ln() / (self.max_neighbors as f32).ln()).floor() as u32)
            .min(self.max_layer as u32) as u8
    }

    fn select_neighbors_heuristic(
        &self,
        candidates: &[PointAndDistance],
        num_neighbors: usize,
    ) -> Vec<PointAndDistance> {
        let mut working_list = BinaryHeap::new();
        let mut return_list: Vec<PointAndDistance> = vec![];
        // Construct a min heap
        for candidate in candidates {
            working_list.push(PointAndDistance {
                point_id: candidate.point_id,
                distance: NotNan::new(-candidate.distance.into_inner()).unwrap(),
            });
        }

        while !working_list.is_empty() && return_list.len() < num_neighbors {
            let e = working_list.pop().unwrap();
            let distance_e_q = -e.distance.into_inner();
            let e_id = e.point_id;
            let mut good = true;
            for x_id in return_list.iter() {
                let distance_x_e = self.distance_two_points(e_id, (*x_id).point_id);
                if distance_x_e < distance_e_q {
                    good = false;
                    break;
                }
            }
            if good {
                return_list.push(PointAndDistance {
                    point_id: e_id,
                    distance: NotNan::new(e.distance.into_inner()).unwrap(),
                });
            }
        }

        return_list
    }

    #[allow(dead_code)]
    fn print(&self) {
        println!("Layers:");
        for (i, layer) in self.layers.iter().enumerate() {
            println!("=== Layer: {} ===", i);
            for (point_id, edges) in layer.edges.iter() {
                println!("Point: {:?}", point_id);
                for edge in edges.iter() {
                    println!("Edge: {:?}", edge);
                }
            }
        }
    }

    #[allow(dead_code)]
    fn validate(&self) -> bool {
        // Traverse layers in reverse order
        let mut current_layer: i32 = self.layers.len() as i32 - 1;
        while current_layer >= 0 {
            let layer = &self.layers[current_layer as usize];

            // Check that if a point is in this layer, it is in lower layers as well
            for (point_id, _) in layer.edges.iter() {
                let mut layer_to_check = current_layer as i32 - 1;
                while layer_to_check >= 0 {
                    let layer_to_check_edges = &self.layers[layer_to_check as usize].edges;
                    if !layer_to_check_edges.contains_key(point_id) {
                        return false;
                    }
                    layer_to_check -= 1;
                }
            }
            current_layer -= 1;
        }

        true
    }
}

impl GraphTraversal for HnswBuilder {
    fn distance(&self, query: &[u8], point_id: u32) -> f32 {
        self.quantizer.distance(query, self.get_vector(point_id))
    }

    fn get_edges_for_point(&self, point_id: u32, layer: u8) -> Option<Vec<u32>> {
        let layer = &self.layers[layer as usize];
        if !layer.edges.contains_key(&point_id) {
            return None;
        }
        Some(layer.edges[&point_id].iter().map(|x| x.point_id).collect())
    }
}

// Test
#[cfg(test)]
mod tests {
    use quantization::pq::ProductQuantizer;

    use super::*;

    fn generate_random_vector(dimension: usize) -> Vec<f32> {
        let mut rng = rand::thread_rng();
        let mut vector = vec![];
        for _ in 0..dimension {
            vector.push(rng.gen::<f32>());
        }
        vector
    }

    #[test]
    fn test_search_layer() {
        let dimension = 10;
        let subvector_dimension = 2;
        let num_bits = 1;

        let mut codebook = vec![];
        for subvector_idx in 0..dimension / subvector_dimension {
            for i in 0..(1 << 1) {
                let x = (subvector_idx * 2 + i) as f32;
                let y = (subvector_idx * 2 + i) as f32;
                codebook.push(x);
                codebook.push(y);
            }
        }
        // Create a temp directory
        let temp_dir = tempdir::TempDir::new("product_quantizer_test").unwrap();
        let base_directory = temp_dir.path().to_str().unwrap().to_string();

        let pq = ProductQuantizer::new(
            dimension,
            subvector_dimension,
            num_bits,
            codebook,
            base_directory.clone(),
            "test_codebook".to_string(),
        );

        let mut builder = HnswBuilder::new(5, 10, 20, Box::new(pq));

        for i in 0..100 {
            builder.insert(i, &generate_random_vector(dimension));
        }

        assert!(builder.validate());
    }
}
