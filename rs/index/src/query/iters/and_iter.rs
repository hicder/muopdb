use crate::query::iters::{InvertedIndexIter, Iter};

/// `AndIter` yields the intersection of all its child iterators.
///
/// It only yields document IDs that are present in **every** child iterator.
///
/// - If any child iterator is exhausted, the AND iterator is exhausted.
/// - Advances all children in lockstep to find common IDs.
///
/// Example:
///   If children yield [1, 3, 5], [3, 5, 7], [3, 4, 5],
///   then AndIter yields [3, 5].
///
/// Used to answer queries like "find documents that match all these conditions".
pub struct AndIter<'a> {
    pub iters: Vec<Iter<'a>>,
}

impl<'a> AndIter<'a> {
    pub fn new(iters: Vec<Iter<'a>>) -> Self {
        Self { iters }
    }
}

impl<'a> InvertedIndexIter for AndIter<'a> {
    /// Advances all child iterators in lockstep and returns the next document ID present in all children (the intersection), or None if any child is exhausted.
    fn next(&mut self) -> Option<u128> {
        todo!()
    }

    /// Advances all child iterators to at least the given doc_id.
    ///
    /// After calling, all children will be positioned at or after doc_id, or exhausted.
    fn skip_to(&mut self, _doc_id: u128) {
        todo!()
    }

    /// Returns the current document ID if all child iterators are positioned at the same doc_id, otherwise None.
    fn doc_id(&self) -> Option<u128> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(ids: &[u128]) -> Iter {
        Iter::Ids(crate::query::iters::ids_iter::IdsIter::new(ids.to_vec()))
    }

    #[test]
    fn test_and_iter_basic_intersection() {
        // [1, 3, 5, 7] ∩ [3, 4, 5, 7, 8] ∩ [2, 3, 5, 7, 9] = [3, 5, 7]
        let a = ids(&[1, 3, 5, 7]);
        let b = ids(&[3, 4, 5, 7, 8]);
        let c = ids(&[2, 3, 5, 7, 9]);
        let mut iter = AndIter::new(vec![a, b, c]);
        let mut results = Vec::new();
        while let Some(doc) = iter.next() {
            results.push(doc);
        }
        assert_eq!(results, vec![3, 5, 7]);
    }

    #[test]
    fn test_and_iter_empty() {
        // Intersection with empty child is empty
        let a = ids(&[1, 2, 3]);
        let b = ids(&[]);
        let mut iter = AndIter::new(vec![a, b]);
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_and_iter_single_child() {
        let a = ids(&[1, 2, 3]);
        let mut iter = AndIter::new(vec![a]);
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next(), Some(2));
        assert_eq!(iter.next(), Some(3));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_and_iter_all_empty() {
        let mut iter = AndIter::new(vec![]);
        assert_eq!(iter.next(), None);
    }
}
