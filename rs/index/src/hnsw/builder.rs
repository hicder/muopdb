use std::cmp::min;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

use anyhow::{anyhow, Context};
use ordered_float::NotNan;
use quantization::quantization::Quantizer;
use rand::Rng;
use utils::l2::L2DistanceCalculatorImpl::StreamingWithSIMD;

use super::utils::{GraphTraversal, PointAndDistance, SearchContext};

/// TODO(hicder): support bare vector in addition to quantized one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layer {
    pub edges: HashMap<u32, Vec<PointAndDistance>>,
}

impl Layer {
    // Reindex the edges in place based on the id mapping
    fn reindex(&mut self, id_mapping: &[i32]) -> anyhow::Result<()> {
        // Drain so we don't create extra Vec
        let tmp_edges = self.edges.drain().collect::<Vec<_>>();
        for (point_id, mut edges) in tmp_edges {
            let new_point_id = id_mapping.get(point_id as usize).ok_or(anyhow!(
                "point_id {} is larger than size of mapping",
                point_id
            ))?;
            for edge in edges.iter_mut() {
                let new_point_id = id_mapping.get(edge.point_id as usize).ok_or(anyhow!(
                    "point_id {} is larger than size of mapping",
                    edge.point_id
                ))?;
                edge.point_id = *new_point_id as u32;
            }
            self.edges.insert(*new_point_id as u32, edges);
        }
        anyhow::Ok(())
    }
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
    pub doc_id_mapping: Vec<u64>,
}

// TODO(hicder): support bare vector in addition to quantized one.
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
            doc_id_mapping: Vec::new(),
        }
    }

    pub fn save_vector(&mut self, vector: &[u8], point_id: u32) {
        // Check if vectors are large enough. If not extend the vectors
        if self.vectors.len() < point_id as usize + 1 {
            self.vectors.resize(point_id as usize + 1, vec![]);
        }
        self.vectors[point_id as usize] = vector.to_vec();
    }

    fn generate_id(&mut self, doc_id: u64) -> u32 {
        let generated_id = self.doc_id_mapping.len() as u32;
        self.doc_id_mapping.push(doc_id);
        return generated_id;
    }

    // Reindex the vectors based on the bottom layer
    // Vectors that are connected should have their id close to each other
    // This optimization is useful for disk-based indexing
    pub fn reindex(&mut self) -> anyhow::Result<()> {
        // Assign new ids to the vectors based on BFS on the bottom layer
        // Assuming our vectors are index from 0 to n-1
        let graph = &mut self.layers[0];
        let vector_length = self.vectors.len();
        // If a vector is visited, it means it's already assigned an id
        // its id should be non-negative
        let mut assigned_ids = vec![-1; vector_length];
        let mut current_id = 0;
        let mut queue: VecDeque<u32> = VecDeque::with_capacity(vector_length);
        for i in 0..vector_length {
            if assigned_ids[i] >= 0 {
                continue;
            }
            queue.push_back(i as u32);
            assigned_ids[i] = current_id;
            current_id += 1;
            while let Some(node) = queue.pop_front() {
                let edges = graph.edges.get_mut(&node);
                if let Some(edges) = edges {
                    edges.sort_by_key(|x| x.distance);
                    for edge in edges {
                        if *assigned_ids.get(edge.point_id as usize).ok_or(anyhow!(
                            "point_id {} is larger than size of vectors",
                            edge.point_id
                        ))? >= 0
                        {
                            continue;
                        }
                        queue.push_back(edge.point_id);
                        assigned_ids[edge.point_id as usize] = current_id;
                        current_id += 1;
                    }
                }
            }
        }
        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer
                .reindex(&assigned_ids)
                .context(format!("failed to reindex layer {}", i))?
        }
        let tmp_id_provider = self.doc_id_mapping.clone();
        for (id, doc_id) in tmp_id_provider.into_iter().enumerate() {
            let new_id = assigned_ids.get(id).ok_or(anyhow!(
                "id in id_provider {} is larger than size of vectors",
                id
            ))?;
            self.doc_id_mapping[*new_id as usize] = doc_id;
        }
        for entry in self.entry_point.iter_mut() {
            let new_id = assigned_ids.get(*entry as usize).ok_or(anyhow!(
                "entrypoint id {} is larger than size of vectors",
                *entry
            ))?;
            *entry = *new_id as u32;
        }
        anyhow::Ok(())
    }

    /// Insert a vector into the index
    pub fn insert(&mut self, doc_id: u64, vector: &[f32]) {
        let mut context = SearchContext::new();
        let quantized_query = self.quantizer.quantize(vector);
        let point_id = self.generate_id(doc_id);

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
        self.quantizer
            .distance(a_vector, b_vector, StreamingWithSIMD)
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

    #[cfg(test)]
    pub fn vectors(&mut self) -> &mut Vec<Vec<u8>> {
        &mut self.vectors
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
        self.quantizer
            .distance(query, self.get_vector(point_id), StreamingWithSIMD)
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
    fn test_hnsw_builder_reindex() {
        let id_provider = vec![100, 101, 102];
        let edges = HashMap::from([
            (
                0,
                vec![PointAndDistance {
                    point_id: 2,
                    distance: NotNan::new(1.0).unwrap(),
                }],
            ),
            (
                1,
                vec![PointAndDistance {
                    point_id: 2,
                    distance: NotNan::new(2.0).unwrap(),
                }],
            ),
            (
                2,
                vec![
                    PointAndDistance {
                        point_id: 1,
                        distance: NotNan::new(2.0).unwrap(),
                    },
                    PointAndDistance {
                        point_id: 0,
                        distance: NotNan::new(1.0).unwrap(),
                    },
                ],
            ),
        ]);
        let expected_mapping = vec![0, 2, 1];
        let layer = Layer { edges };

        let mut codebook = vec![];
        for subvector_idx in 0..5 {
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
        let mut builder = HnswBuilder {
            vectors: vec![vec![]; 3],
            max_neighbors: 1,
            layers: vec![layer],
            current_top_layer: 0,
            quantizer: Box::new(ProductQuantizer::new(
                10,
                2,
                1,
                codebook,
                base_directory.clone(),
            )),
            ef_contruction: 0,
            entry_point: vec![0, 1],
            max_layer: 0,
            doc_id_mapping: id_provider,
        };
        builder.reindex().unwrap();
        for i in 0..3 {
            assert_eq!(
                builder
                    .doc_id_mapping
                    .get(expected_mapping[i] as usize)
                    .unwrap(),
                &((i + 100) as u64)
            );
        }
        assert_eq!(builder.entry_point, vec![0, 2]);
        assert_eq!(
            builder.layers[0].edges.get(&0).unwrap(),
            &vec![PointAndDistance {
                point_id: 1,
                distance: NotNan::new(1.0).unwrap(),
            }]
        );
        assert_eq!(
            builder.layers[0].edges.get(&1).unwrap(),
            &vec![
                PointAndDistance {
                    point_id: 0,
                    distance: NotNan::new(1.0).unwrap(),
                },
                PointAndDistance {
                    point_id: 2,
                    distance: NotNan::new(2.0).unwrap(),
                }
            ]
        );
        assert_eq!(
            builder.layers[0].edges.get(&2).unwrap(),
            &vec![PointAndDistance {
                point_id: 1,
                distance: NotNan::new(2.0).unwrap(),
            }]
        );
    }
    #[test]
    fn test_layer_reindex() {
        let mut edges = HashMap::new();
        for i in 0..10 {
            edges.insert(
                i,
                vec![
                    PointAndDistance {
                        point_id: (i + 1) % 10,
                        distance: NotNan::new(1.0).unwrap(),
                    },
                    PointAndDistance {
                        point_id: (i + 5) % 10,
                        distance: NotNan::new(2.0).unwrap(),
                    },
                ],
            );
        }
        let og_layer = Layer { edges };
        let mut layer = og_layer.clone();
        // Similar mapping do not change
        layer.reindex(&vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).unwrap();
        assert_eq!(layer, og_layer);

        // Reverse mapping
        layer.reindex(&vec![9, 8, 7, 6, 5, 4, 3, 2, 1, 0]).unwrap();
        for i in 0..10 {
            let expected_edges = vec![
                PointAndDistance {
                    point_id: ((i as i32 - 1 + 10) % 10) as u32,
                    distance: NotNan::new(1.0).unwrap(),
                },
                PointAndDistance {
                    point_id: ((i as i32 - 5 + 10) % 10) as u32,
                    distance: NotNan::new(2.0).unwrap(),
                },
            ];
            assert_eq!(layer.edges.get(&i).unwrap(), &expected_edges);
        }
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
        );

        let mut builder = HnswBuilder::new(5, 10, 20, Box::new(pq));

        for i in 0..100 {
            builder.insert(i, &generate_random_vector(dimension));
        }

        assert!(builder.validate());
    }
}
