pub mod and_iter;
pub mod ids_iter;
pub mod or_iter;
pub mod term_iter;

use and_iter::AndIter;
use ids_iter::IdsIter;
use or_iter::OrIter;
use term_iter::TermIter;

/// State of the iterator: NotStarted, At(data), or Exhausted
/// Used internally by various iterators to track their position.
/// T represents the state data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IterState<T> {
    NotStarted,
    At(T),
    Exhausted,
}

/// A unified enum to represent all types of iterators that implement InvertedIndexIter trait.
pub enum Iter<'a> {
    And(AndIter<'a>),
    Or(OrIter<'a>),
    Ids(IdsIter),
    Term(TermIter<'a>),
}

impl<'a> InvertedIndexIter for Iter<'a> {
    fn next(&mut self) -> Option<u32> {
        match self {
            Iter::And(iter) => iter.next(),
            Iter::Or(iter) => iter.next(),
            Iter::Ids(iter) => iter.next(),
            Iter::Term(iter) => iter.next(),
        }
    }

    fn skip_to(&mut self, point_id: u32) {
        match self {
            Iter::And(iter) => iter.skip_to(point_id),
            Iter::Or(iter) => iter.skip_to(point_id),
            Iter::Ids(iter) => iter.skip_to(point_id),
            Iter::Term(iter) => iter.skip_to(point_id),
        }
    }

    fn point_id(&mut self) -> Option<u32> {
        match self {
            Iter::And(iter) => iter.point_id(),
            Iter::Or(iter) => iter.point_id(),
            Iter::Ids(iter) => iter.point_id(),
            Iter::Term(iter) => iter.point_id(),
        }
    }
}

pub trait InvertedIndexIter {
    fn next(&mut self) -> Option<u32>;

    fn skip_to(&mut self, point_id: u32);

    fn point_id(&mut self) -> Option<u32>;
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::query::iters::and_iter::AndIter;
    use crate::query::iters::ids_iter::IdsIter;
    use crate::query::iters::or_iter::OrIter;

    fn ids(ids: &[u32]) -> Iter<'_> {
        Iter::Ids(IdsIter::new(ids.to_vec()))
    }

    #[test]
    fn test_or_and_ids_nested() {
        // ( [1, 2, 3, 4, 5] ∩ [3, 4, 5, 6, 7] ) ∪ [4, 5, 6, 7, 8]
        // AndIter yields: [3, 4, 5]
        // Union with [4, 5, 6, 7, 8] yields: [3, 4, 5, 6, 7, 8]
        let a = ids(&[1, 2, 3, 4, 5]);
        let b = ids(&[3, 4, 5, 6, 7]);
        let c = ids(&[4, 5, 6, 7, 8]);
        let and = Iter::And(AndIter::new(vec![a, b]));
        let mut iter = OrIter::new(vec![and, c]);
        let mut results = Vec::new();
        while let Some(point) = iter.next() {
            results.push(point);
        }
        assert_eq!(results, vec![3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn test_and_or_ids_nested() {
        // ( [1, 2, 3, 4, 5] ∪ [3, 4, 5, 6, 7] ) ∩ [4, 5, 6, 7, 8]
        // OrIter yields: [1, 2, 3, 4, 5, 6, 7]
        // Intersection with [4, 5, 6, 7, 8] yields: [4, 5, 6, 7]
        let a = ids(&[1, 2, 3, 4, 5]);
        let b = ids(&[3, 4, 5, 6, 7]);
        let c = ids(&[4, 5, 6, 7, 8]);
        let or = Iter::Or(OrIter::new(vec![a, b]));
        let and = AndIter::new(vec![or, c]);
        let mut iter = and;
        let mut results = Vec::new();
        while let Some(point) = iter.next() {
            results.push(point);
        }
        assert_eq!(results, vec![4, 5, 6, 7]);
    }

    #[test]
    fn test_and_of_ors() {
        // ([1, 2, 3] ∪ [2, 3, 4]) ∩ ([2, 3, 5] ∪ [3, 5, 6])
        // Left Or: [1, 2, 3, 4]
        // Right Or: [2, 3, 5, 6]
        // Intersection: [2, 3]
        let left = Iter::Or(OrIter::new(vec![ids(&[1, 2, 3]), ids(&[2, 3, 4])]));
        let right = Iter::Or(OrIter::new(vec![ids(&[2, 3, 5]), ids(&[3, 5, 6])]));
        let mut iter = AndIter::new(vec![left, right]);
        let mut results = Vec::new();
        while let Some(point) = iter.next() {
            results.push(point);
        }
        assert_eq!(results, vec![2, 3]);
    }

    #[test]
    fn test_or_of_ands() {
        // ([1, 2, 3] ∩ [2, 3, 4]) ∪ ([2, 3, 5] ∩ [3, 5, 6])
        // Left And: [2, 3]
        // Right And: [3, 5]
        // Union: [2, 3, 5]
        let left = Iter::And(AndIter::new(vec![ids(&[1, 2, 3]), ids(&[2, 3, 4])]));
        let right = Iter::And(AndIter::new(vec![ids(&[2, 3, 5]), ids(&[3, 5, 6])]));
        let mut iter = OrIter::new(vec![left, right]);
        let mut results = Vec::new();
        while let Some(point) = iter.next() {
            results.push(point);
        }
        assert_eq!(results, vec![2, 3, 5]);
    }

    #[test]
    fn test_deeply_nested_combinations() {
        // ((([1, 2, 3] ∩ [2, 3, 4]) ∪ [5, 6]) ∩ ([2, 3, 5] ∪ [3, 5, 6]))
        // [1,2,3] ∩ [2,3,4] = [2,3]
        // ([2,3] ∪ [5,6]) = [2,3,5,6]
        // [2,3,5,6] ∩ ([2,3,5] ∪ [3,5,6]) = [2,3,5,6]
        // ([2,3,5] ∪ [3,5,6]) = [2,3,5,6]
        // Final: [2,3,5,6]
        let left_and = Iter::And(AndIter::new(vec![ids(&[1, 2, 3]), ids(&[2, 3, 4])]));
        let left_or = Iter::Or(OrIter::new(vec![left_and, ids(&[5, 6])]));
        let right_or = Iter::Or(OrIter::new(vec![ids(&[2, 3, 5]), ids(&[3, 5, 6])]));
        let mut iter = AndIter::new(vec![left_or, right_or]);
        let mut results = Vec::new();
        while let Some(point) = iter.next() {
            results.push(point);
        }
        assert_eq!(results, vec![2, 3, 5, 6]);
    }
}
