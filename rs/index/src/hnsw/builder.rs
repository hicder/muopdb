use std::cmp::min;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::vec;

use anyhow::{anyhow, Context, Result};
use bit_vec::BitVec;
use log::debug;
use ordered_float::NotNan;
use quantization::quantization::Quantizer;
use quantization::typing::VectorOps;
use rand::Rng;

use super::index::Hnsw;
use super::utils::{BuilderContext, GraphTraversal};
use crate::utils::PointAndDistance;
use crate::vector::file::FileBackedAppendableVectorStorage;
use crate::vector::VectorStorageConfig;

/// TODO(hicder): support bare vector in addition to quantized one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layer {
    pub edges: HashMap<u32, Vec<PointAndDistance>>,
}

impl Layer {
    // Reindex the edges in place based on the id mapping
    fn reindex(&mut self, id_mapping: &[i32]) -> Result<()> {
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
        Ok(())
    }
}

/// The actual builder
pub struct HnswBuilder<Q: Quantizer> {
    vectors: FileBackedAppendableVectorStorage<Q::QuantizedT>,

    max_neighbors: usize,
    pub layers: Vec<Layer>,
    pub current_top_layer: u8,
    pub quantizer: Q,
    ef_contruction: u32,
    pub entry_point: Vec<u32>,
    max_layer: u8,
    pub doc_id_mapping: Vec<u128>,
}

// TODO(hicder): support bare vector in addition to quantized one.
impl<Q: Quantizer> HnswBuilder<Q> {
    pub fn new(
        max_neighbors: usize,
        max_layers: u8,
        ef_construction: u32,
        vector_storage_memory_size: usize,
        vector_storage_file_size: usize,
        num_features: usize,
        quantizer: Q,
        base_directory: String,
    ) -> Self {
        let vectors = FileBackedAppendableVectorStorage::<Q::QuantizedT>::new(
            base_directory.clone(),
            vector_storage_memory_size,
            vector_storage_file_size,
            num_features,
        );

        Self {
            vectors,
            max_neighbors,
            max_layer: max_layers,
            layers: vec![],
            current_top_layer: 0,
            quantizer,
            ef_contruction: ef_construction,
            entry_point: vec![],
            doc_id_mapping: Vec::new(),
        }
    }

    pub fn from_hnsw(
        hnsw: Hnsw<Q>,
        output_directory: String,
        vector_storage_config: VectorStorageConfig,
        max_neighbors: usize,
    ) -> Self {
        let num_layers = hnsw.get_header().num_layers as usize;
        let mut current_top_layer = (num_layers - 1) as u8;

        let mut layers = vec![];
        for _ in 0..num_layers {
            layers.push(Layer {
                edges: HashMap::new(),
            });
        }

        let tmp_vector_storage_dir = format!("{}/vector_storage_tmp", output_directory);
        let mut vector_storage = FileBackedAppendableVectorStorage::<Q::QuantizedT>::new(
            tmp_vector_storage_dir,
            vector_storage_config.memory_threshold,
            vector_storage_config.file_size,
            vector_storage_config.num_features,
        );

        // Copy over the vectors
        for i in 0..hnsw.get_doc_id_mapping_slice().len() {
            let vector = hnsw.vector_storage.get_no_context(i as u32).unwrap();
            vector_storage
                .append(vector)
                .unwrap_or_else(|_| panic!("append failed"));
        }

        loop {
            let layer = &mut layers[current_top_layer as usize];
            hnsw.visit(current_top_layer, |from: u32, to: u32| {
                let from_v = hnsw.vector_storage.get_no_context(from as u32).unwrap();
                let to_v = hnsw.vector_storage.get_no_context(to as u32).unwrap();
                let distance = Q::QuantizedT::distance(from_v, to_v, &hnsw.quantizer);
                layer
                    .edges
                    .entry(from)
                    .or_insert_with(|| vec![])
                    .push(PointAndDistance {
                        point_id: to,
                        distance: NotNan::new(distance).unwrap(),
                    });
                true
            });

            debug!(
                "Layer {}, number of edges: {}",
                current_top_layer,
                layer.edges.len()
            );
            if current_top_layer == 0 {
                break;
            }
            current_top_layer -= 1;
        }

        let all_entry_points = hnsw.get_all_entry_points();
        let doc_id_mapping = hnsw.get_doc_id_mapping_slice().to_vec();

        Self {
            vectors: vector_storage,
            max_neighbors: max_neighbors,
            max_layer: num_layers as u8,
            layers: layers,
            current_top_layer: num_layers as u8 - 1,
            quantizer: hnsw.quantizer,
            ef_contruction: 100,
            entry_point: all_entry_points,
            doc_id_mapping: doc_id_mapping,
        }
    }

    fn generate_id(&mut self, doc_id: u128) -> u32 {
        let generated_id = self.doc_id_mapping.len() as u32;
        self.doc_id_mapping.push(doc_id);
        return generated_id;
    }

    pub fn reindex_layer(
        &mut self,
        layer: u8,
        assigned_ids: &mut Vec<i32>,
        current_id: &mut i32,
        vector_length: usize,
    ) -> Result<()> {
        debug!("Reindex layer {}", layer);
        let graph = &mut self.layers[layer as usize];
        let mut queue: VecDeque<u32> = VecDeque::with_capacity(vector_length);
        let mut points: Vec<u32> = graph.edges.keys().map(|x| *x).collect();
        points.sort();
        let mut visited: BitVec = BitVec::from_elem(vector_length, false);

        for e in points.iter() {
            if visited.get((*e).try_into().unwrap()).unwrap_or(false) {
                continue;
            }

            queue.push_back(*e);
            if assigned_ids[*e as usize] < 0 {
                assigned_ids[*e as usize] = *current_id;
                *current_id += 1;
            }
            while let Some(node) = queue.pop_front() {
                visited.set(node.try_into().unwrap(), true);

                let edges = graph.edges.get_mut(&node);
                if let Some(edges) = edges {
                    // Try to get the closest edge to be the nearest assigned id.
                    edges.sort_by_key(|x| x.distance);
                    for edge in edges {
                        if visited
                            .get(edge.point_id.try_into().unwrap())
                            .unwrap_or(false)
                        {
                            continue;
                        }
                        queue.push_back(edge.point_id as u32);
                        if assigned_ids[edge.point_id as usize] < 0 {
                            assigned_ids[edge.point_id as usize] = *current_id;
                            *current_id += 1;
                        }
                        visited.set(edge.point_id.try_into().unwrap(), true);
                    }
                }
            }
        }
        Ok(())
    }

    /// Assign new ids to the vectors based on BFS on all layers
    fn get_reassigned_ids(&mut self) -> Result<Vec<i32>> {
        let vector_length = self.vectors.num_vectors();
        let mut assigned_ids = vec![-1; vector_length];
        let mut current_id = 0;
        let num_layers = self.layers.len();
        for layer in 0..num_layers {
            self.reindex_layer(
                (num_layers - 1 - layer) as u8,
                &mut assigned_ids,
                &mut current_id,
                vector_length,
            )?;
        }
        Ok(assigned_ids)
    }

    // Reindex the vectors based on the bottom layer
    // Vectors that are connected should have their id close to each other
    // This optimization is useful for disk-based indexing
    pub fn reindex(&mut self, temp_dir: String) -> Result<()> {
        let assigned_ids = self.get_reassigned_ids()?;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer
                .reindex(&assigned_ids)
                .context(format!("failed to reindex layer {}", i))?;
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
        self.entry_point.sort();

        // Build reverse assigned ids
        let mut reverse_assigned_ids = vec![-1; self.doc_id_mapping.len()];
        for (i, id) in assigned_ids.iter().enumerate() {
            reverse_assigned_ids[*id as usize] = i as i32;
        }

        let vector_storage_config = self.vectors.config();
        let mut new_vector_storage = FileBackedAppendableVectorStorage::<Q::QuantizedT>::new(
            temp_dir.clone(),
            vector_storage_config.memory_threshold,
            vector_storage_config.file_size,
            vector_storage_config.num_features,
        );
        for i in 0..reverse_assigned_ids.len() {
            let mapped_id = reverse_assigned_ids[i];
            let vector = self.vectors.get_no_context(mapped_id as u32).unwrap();
            new_vector_storage
                .append(vector)
                .unwrap_or_else(|_| panic!("append failed"));
        }

        self.vectors = new_vector_storage;
        Ok(())
    }

    /// Insert a vector into the index
    pub fn insert(&mut self, doc_id: u128, vector: &[f32]) -> Result<()> {
        let quantized_query = Q::QuantizedT::process_vector(vector, &self.quantizer);
        let point_id = self.generate_id(doc_id);
        let mut context = BuilderContext::new(point_id + 1);

        let empty_graph = point_id == 0;
        self.vectors.append(&quantized_query)?;
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
            return Ok(());
        }

        let mut entry_point = self.entry_point[0];
        if layer < self.current_top_layer {
            for l in ((layer + 1)..=self.current_top_layer).rev() {
                let nearest_elements =
                    self.search_layer(&mut context, &quantized_query, entry_point, 1, l);
                entry_point = nearest_elements[0].point_id as u32;
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
        Ok(())
    }

    pub fn get_nodes_from_non_bottom_layer(&self) -> Vec<u32> {
        let mut nodes = HashSet::new();
        let mut current_layer = self.current_top_layer;
        while current_layer > 0 {
            debug!("get nodes from layer: {:?}", current_layer);
            for (point_id, _) in self.layers[current_layer as usize].edges.iter() {
                nodes.insert(*point_id);
            }
            current_layer -= 1;
        }
        nodes.into_iter().collect()
    }

    /// Compute the distance between two points.
    /// The vectors are quantized.
    fn distance_two_points(&self, a: u32, b: u32) -> f32 {
        let a_vector = self.get_vector(a);
        let b_vector = self.get_vector(b);
        Q::QuantizedT::distance(a_vector, b_vector, &self.quantizer)
    }

    fn get_vector(&self, point_id: u32) -> &[Q::QuantizedT] {
        self.vectors.get_no_context(point_id).unwrap()
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

    pub fn vectors(&mut self) -> &mut FileBackedAppendableVectorStorage<Q::QuantizedT> {
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

impl<Q: Quantizer> GraphTraversal<Q> for HnswBuilder<Q> {
    type ContextT = BuilderContext;

    fn distance(
        &self,
        query: &[Q::QuantizedT],
        point_id: u32,
        _context: &mut BuilderContext,
    ) -> f32 {
        let point = self.vectors.get(point_id, _context).unwrap();
        Q::QuantizedT::distance(query, point, &self.quantizer)
    }

    fn get_edges_for_point(&self, point_id: u32, layer: u8) -> Option<Vec<u32>> {
        let layer = &self.layers[layer as usize];
        if !layer.edges.contains_key(&point_id) {
            return None;
        }
        Some(layer.edges[&point_id].iter().map(|x| x.point_id).collect())
    }

    fn print_graph(&self, _layer: u8, _predicate: impl Fn(u8, u32) -> bool) {
        // TODO
        return;
    }
}

// Test
#[cfg(test)]
mod tests {
    use std::fs;

    use quantization::pq::pq::ProductQuantizer;
    use utils::distance::l2::L2DistanceCalculator;
    use utils::test_utils::generate_random_vector;

    use super::*;
    use crate::vector::file::FileBackedAppendableVectorStorage;

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
        let vector_dir = format!("{}/vectors", base_directory);
        fs::create_dir_all(vector_dir.clone()).unwrap();
        let mut vectors = FileBackedAppendableVectorStorage::<u8>::new(vector_dir, 1024, 4096, 5);
        vectors.append(&vec![0, 0, 0, 0, 0]).unwrap();
        vectors.append(&vec![1, 1, 1, 1, 1]).unwrap();
        vectors.append(&vec![2, 2, 2, 2, 2]).unwrap();

        let mut builder = HnswBuilder {
            vectors,
            max_neighbors: 1,
            layers: vec![layer],
            current_top_layer: 0,
            quantizer: ProductQuantizer::<L2DistanceCalculator>::new(
                10,
                2,
                1,
                codebook,
                base_directory.clone(),
            )
            .expect("Can't create product quantizer"),
            ef_contruction: 0,
            entry_point: vec![0, 1],
            max_layer: 0,
            doc_id_mapping: id_provider,
        };
        builder.reindex(base_directory.clone()).unwrap();

        for i in 0..3 {
            assert!(
                *builder
                    .doc_id_mapping
                    .get(expected_mapping[i] as usize)
                    .unwrap()
                    >= 100
            );
            assert!(
                *builder
                    .doc_id_mapping
                    .get(expected_mapping[i] as usize)
                    .unwrap()
                    <= 102
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
        )
        .expect("ProductQuantizer should be created.");

        let vector_dir = format!("{}/vectors", base_directory);
        fs::create_dir_all(vector_dir.clone()).unwrap();
        let mut builder = HnswBuilder::<ProductQuantizer<L2DistanceCalculator>>::new(
            5, 10, 20, 1024, 4096, 5, pq, vector_dir,
        );

        for i in 0..100 {
            builder
                .insert(i, &generate_random_vector(dimension))
                .unwrap();
        }

        assert!(builder.validate());
    }
}
