use std::cmp::{Ord, Ordering};
use std::collections::HashSet;

use ordered_float::NotNan;
use roaring::RoaringBitmap;

use crate::vector::StorageContext;

pub struct SearchContext {
    pub visited: RoaringBitmap,
    pub record_pages: bool,
    pub visited_pages: Option<HashSet<String>>,
}

impl SearchContext {
    pub fn new(record_pages: bool) -> Self {
        if !record_pages {
            Self {
                visited: RoaringBitmap::new(),
                record_pages: false,
                visited_pages: None,
            }
        } else {
            Self {
                visited: RoaringBitmap::new(),
                record_pages: true,
                visited_pages: Some(HashSet::new()),
            }
        }
    }

    pub fn num_pages_accessed(&self) -> usize {
        if !self.record_pages {
            return 0;
        }

        self.visited_pages.as_ref().unwrap().len()
    }
}

impl StorageContext for SearchContext {
    fn should_record_pages(&self) -> bool {
        self.record_pages
    }

    fn record_pages(&mut self, page_id: String) {
        if let Some(visited_pages) = &mut self.visited_pages {
            visited_pages.insert(page_id);
        }
    }

    fn num_pages_accessed(&self) -> usize {
        0
    }

    fn reset_pages_accessed(&mut self) {}

    fn set_visited(&mut self, _id: u32) {
        self.visited.insert(_id);
    }

    fn visited(&self, _id: u32) -> bool {
        self.visited.contains(_id)
    }
}

/// PointAndDistance is used to store the distance between a point and a query
/// This is only meaningful inside a segment, since point_id is not unique within a segment,
/// but is unique within a segment. To return to the user, we need to map point_id to
/// the actual doc_id
#[derive(PartialEq, Eq, Ord, PartialOrd, Clone, Debug)]
pub struct PointAndDistance {
    pub distance: NotNan<f32>,
    pub point_id: u32,
}

impl PointAndDistance {
    pub fn new(distance: f32, point_id: u32) -> Self {
        PointAndDistance {
            distance: NotNan::new(distance).unwrap(),
            point_id,
        }
    }
}

/// IdWithScore is used to store the doc_id and score of a document
/// This is only meaningful across segments, since doc_id is unique across segments
/// This is the id that will be returned to the user
#[derive(Debug)]
pub struct IdWithScore {
    pub doc_id: u128,
    pub score: f32,
}

impl Ord for IdWithScore {
    fn cmp(&self, other: &Self) -> Ordering {
        // Handle NaN cases first
        if self.score.is_nan() && other.score.is_nan() {
            self.doc_id.cmp(&other.doc_id) // Both are NaN, consider them equal, tie-break by id
        } else if self.score.is_nan() {
            Ordering::Greater // This instance is NaN, consider it greater
        } else if other.score.is_nan() {
            Ordering::Less // Other instance is NaN, consider it less
        } else {
            match self.score.partial_cmp(&other.score) {
                // Tie-break by id
                Some(Ordering::Equal) => self.doc_id.cmp(&other.doc_id),
                Some(order) => order,
                // Handle unexpected cases (shouldn't happen with valid inputs)
                None => Ordering::Equal,
            }
        }
    }
}

impl PartialOrd for IdWithScore {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for IdWithScore {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.doc_id == other.doc_id
    }
}

impl Eq for IdWithScore {}

pub struct SearchStats {
    pub num_pages_accessed: usize,
}

impl Default for SearchStats {
    fn default() -> Self {
        Self::new()
    }
}

impl SearchStats {
    pub fn new() -> Self {
        Self {
            num_pages_accessed: 0,
        }
    }

    pub fn merge(&mut self, other: &Self) {
        self.num_pages_accessed += other.num_pages_accessed;
    }
}

pub struct SearchResult {
    pub id_with_scores: Vec<IdWithScore>,
    pub stats: SearchStats,
}

unsafe impl Send for SearchResult {}

impl Default for SearchResult {
    fn default() -> Self {
        Self::new()
    }
}

impl SearchResult {
    pub fn new() -> Self {
        Self {
            id_with_scores: Vec::new(),
            stats: SearchStats::new(),
        }
    }
}
pub struct IntermediateResult {
    pub point_and_distances: Vec<PointAndDistance>,
    pub stats: SearchStats,
}

mod tests {
    #[cfg(test)]
    use crate::utils::IdWithScore;

    #[test]
    fn test_id_with_score_ord() {
        let a = IdWithScore {
            doc_id: 2,
            score: 1.0,
        };
        let b = IdWithScore {
            doc_id: 1,
            score: 2.0,
        };
        let c = IdWithScore {
            doc_id: 1,
            score: 1.0,
        };
        let d = IdWithScore {
            doc_id: 2,
            score: 1.0,
        };
        let e = IdWithScore {
            doc_id: 1,
            score: 1.0,
        };

        // Test basic comparison
        assert!(a < b); // a has a smaller score than b
        assert!(b > a); // b has a larger score than a

        // Test tie-breaking by id
        assert!(c < d); // c has the same score as d but a smaller id

        // Test equality
        assert_eq!(c, e); // c and e are equal in terms of score and id

        // Test NaN handling
        let f = IdWithScore {
            doc_id: 3,
            score: f32::NAN,
        };
        let g = IdWithScore {
            doc_id: 4,
            score: f32::NAN,
        };

        assert!(f < g); // f has a smaller id than g
        assert!(a < f); // f is NaN
    }

    #[test]
    fn test_id_with_score_sorting() {
        let mut scores = [
            IdWithScore {
                score: f32::NAN,
                doc_id: 5,
            },
            IdWithScore {
                score: 1.0,
                doc_id: 2,
            },
            IdWithScore {
                score: 1.0,
                doc_id: 1,
            },
            IdWithScore {
                score: 3.0,
                doc_id: 0,
            },
            IdWithScore {
                score: f32::NAN,
                doc_id: 4,
            },
            IdWithScore {
                score: 2.0,
                doc_id: 1,
            },
        ];

        scores.sort();

        let expected_order = [
            IdWithScore {
                score: 1.0,
                doc_id: 1,
            },
            IdWithScore {
                score: 1.0,
                doc_id: 2,
            },
            IdWithScore {
                score: 2.0,
                doc_id: 1,
            },
            IdWithScore {
                score: 3.0,
                doc_id: 0,
            },
            IdWithScore {
                score: f32::NAN,
                doc_id: 4,
            },
            IdWithScore {
                score: f32::NAN,
                doc_id: 5,
            },
        ];

        for (a, b) in scores.iter().zip(expected_order.iter()) {
            if a.score.is_nan() {
                assert!(b.score.is_nan());
                assert_eq!(a.doc_id, b.doc_id);
            } else {
                assert_eq!(a.score, b.score);
                assert_eq!(a.doc_id, b.doc_id);
            }
        }
    }
}
