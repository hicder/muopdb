use std::collections::BinaryHeap;

use bit_vec::BitVec;
use ordered_float::NotNan;

use crate::utils::PointAndDistance;

/// Hold the state of the traversal.
#[derive(Default)]
pub struct TraverseState {
    pub visited: BitVec,
    pub min_heap: BinaryHeap<PointAndDistance>,
}

impl TraverseState {
    pub fn push(&mut self, point_id: u32, distance: f32) {
        self.min_heap.push(PointAndDistance {
            distance: NotNan::new(-distance).unwrap(),
            point_id,
        });
        self.visited.set(point_id as usize, true);
    }

    pub fn is_visited(&self, point_id: u32) -> bool {
        self.visited.get(point_id as usize).unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_hnsw() {
        let mut state = TraverseState::default();
        state.min_heap.push(PointAndDistance {
            distance: NotNan::new(0.0).unwrap(),
            point_id: 0,
        });
        state.min_heap.push(PointAndDistance {
            distance: NotNan::new(-2.0).unwrap(),
            point_id: 2,
        });
        state.min_heap.push(PointAndDistance {
            distance: NotNan::new(-1.0).unwrap(),
            point_id: 1,
        });

        for i in 0..3 {
            let point_and_distance = state.min_heap.pop().unwrap();
            assert_eq!(i, point_and_distance.point_id);
        }
    }
}
