use std::collections::BinaryHeap;

use bit_vec::BitVec;
use ordered_float::NotNan;
use quantization::quantization::Quantizer;

use crate::utils::TraversalContext;
pub struct BuilderContext {
    visited: BitVec,
}

impl BuilderContext {
    pub fn new(max_id: u32) -> Self {
        Self {
            visited: BitVec::from_elem(max_id as usize, false),
        }
    }
}

impl TraversalContext for BuilderContext {
    fn visited(&self, i: u32) -> bool {
        self.visited.get(i as usize).unwrap_or(false)
    }

    fn set_visited(&mut self, i: u32) {
        self.visited.set(i as usize, true);
    }

    fn should_record_pages(&self) -> bool {
        false
    }

    fn record_pages(&mut self, _page_id: String) {}
}

/// A point and its distance to the query.
#[derive(PartialEq, Eq, Ord, PartialOrd, Clone, Debug)]
pub struct PointAndDistance {
    pub point_id: u32,
    pub distance: NotNan<f32>,
}

/// Move the traversal logic out, since it's used in both indexing and query path
pub trait GraphTraversal<Q: Quantizer> {
    type ContextT: TraversalContext;

    /// Distance between the query and point_id
    fn distance(&self, query: &[Q::QuantizedT], point_id: u32, context: &mut Self::ContextT)
        -> f32;

    /// Get the edges for a point
    fn get_edges_for_point(&self, point_id: u32, layer: u8) -> Option<Vec<u32>>;

    fn search_layer(
        &self,
        context: &mut Self::ContextT,
        query: &[Q::QuantizedT],
        entry_point: u32,
        ef_construction: u32,
        layer: u8,
    ) -> Vec<PointAndDistance> {
        // Mark the entry point as visited so that we don't visit it again
        context.set_visited(entry_point);

        // candidate is min heap while working list is max heap
        // TODO(hicder): Probably use the comparator instead of this hack?
        let mut candidates = BinaryHeap::new();
        let mut working_list = BinaryHeap::new();

        candidates.push(PointAndDistance {
            point_id: entry_point,
            distance: NotNan::new(-self.distance(query, entry_point, context)).unwrap(),
        });
        working_list.push(PointAndDistance {
            point_id: entry_point,
            distance: NotNan::new(self.distance(query, entry_point, context)).unwrap(),
        });

        while !candidates.is_empty() {
            let point_and_distance = candidates.pop().unwrap();
            let point_id = point_and_distance.point_id;
            let distance: f32 = -*point_and_distance.distance;

            let mut furthest_element_from_working_list = working_list.peek().unwrap();
            if distance > *furthest_element_from_working_list.distance {
                // All elements in W are evaluated, so we can stop
                break;
            }

            let edges = self.get_edges_for_point(point_id, layer);
            if edges.is_none() {
                continue;
            }

            for e in edges.unwrap().iter() {
                if context.visited(*e) {
                    continue;
                }
                context.set_visited(*e);
                furthest_element_from_working_list = working_list.peek().unwrap();
                let distance_e_q = self.distance(query, *e, context);
                if distance_e_q < *furthest_element_from_working_list.distance
                    || working_list.len() < ef_construction as usize
                {
                    candidates.push(PointAndDistance {
                        point_id: *e,
                        distance: NotNan::new(-distance_e_q).unwrap(),
                    });
                    working_list.push(PointAndDistance {
                        point_id: *e,
                        distance: NotNan::new(distance_e_q).unwrap(),
                    });
                    if working_list.len() > ef_construction as usize {
                        working_list.pop();
                    }
                }
            }
        }

        // Probably should return the distance as well, and let customers decide
        // whether to drop the distance or not
        let mut result: Vec<PointAndDistance> = working_list.into_iter().collect();
        result.sort();
        result
    }

    /// Print the graph for debugging purposes
    fn print_graph(&self, layer: u8, predicate: impl Fn(u8, u32) -> bool);
}
