use crate::query::iters::{InvertedIndexIter, Iter};

/// `OrIter` yields the union of all its child iterators.
///
/// It yields every document ID that appears in any child iterator, without duplicates, in sorted order.
///
/// - If a child iterator is exhausted, it is ignored for the rest of the iteration.
/// - At each step, finds the smallest current doc_id among all children, yields it, and advances all children at that doc_id.
///
/// Example:
///   If children yield [1, 3, 5], [3, 5, 7], [2, 3, 5],
///   then OrIter yields [1, 2, 3, 5, 7].
///
/// Used to answer queries like "find documents that match any of these conditions".
pub struct OrIter<'a> {
    pub iters: Vec<Iter<'a>>,
}

impl<'a> OrIter<'a> {
    pub fn new(iters: Vec<Iter<'a>>) -> Self {
        Self { iters }
    }
}

impl<'a> InvertedIndexIter for OrIter<'a> {
    /// Advances all child iterators and returns the next document ID present in any child (the union), or None if all are exhausted.
    fn next(&mut self) -> Option<u128> {
        todo!()
    }

    /// Advances all child iterators to at least the given doc_id.
    ///
    /// After calling, all children will be positioned at or after doc_id, or exhausted.
    fn skip_to(&mut self, _doc_id: u128) {
        todo!()
    }

    /// Returns the smallest current document ID among all children, or None if all are exhausted.
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
    fn test_or_iter_basic_union() {
        // [1, 3, 5, 7] ∪ [3, 4, 5, 7, 8] ∪ [2, 3, 5, 7, 9] = [1, 2, 3, 4, 5, 7, 8, 9]
        let a = ids(&[1, 3, 5, 7]);
        let b = ids(&[3, 4, 5, 7, 8]);
        let c = ids(&[2, 3, 5, 7, 9]);
        let mut iter = OrIter::new(vec![a, b, c]);
        let mut results = Vec::new();
        while let Some(doc) = iter.next() {
            results.push(doc);
        }
        assert_eq!(results, vec![1, 2, 3, 4, 5, 7, 8, 9]);
    }

    #[test]
    fn test_or_iter_empty() {
        // Union with empty child is just the non-empty child
        let a = ids(&[1, 2, 3]);
        let b = ids(&[]);
        let mut iter = OrIter::new(vec![a, b]);
        let mut results = Vec::new();
        while let Some(doc) = iter.next() {
            results.push(doc);
        }
        assert_eq!(results, vec![1, 2, 3]);
    }

    #[test]
    fn test_or_iter_single_child() {
        let a = ids(&[1, 2, 3]);
        let mut iter = OrIter::new(vec![a]);
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next(), Some(2));
        assert_eq!(iter.next(), Some(3));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_or_iter_all_empty() {
        let mut iter = OrIter::new(vec![]);
        assert_eq!(iter.next(), None);
    }
}
