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
    pub fn new(mut iters: Vec<Iter<'a>>) -> Self {
        // Prime all children: advance if doc_id() is None
        for iter in &mut iters {
            if iter.doc_id().is_none() {
                iter.next();
            }
        }
        Self { iters }
    }
}

impl<'a> InvertedIndexIter for AndIter<'a> {
    /// Advances all child iterators in lockstep and returns the next document ID present in all children (the intersection), or None if any child is exhausted.
    fn next(&mut self) -> Option<u128> {
        loop {
            // Find the maximum doc_id among all children
            let mut max_doc = None;
            for iter in self.iters.iter() {
                // Any exhausted child means the AND is exhausted -> return None
                let doc = iter.doc_id()?;
                max_doc = Some(max_doc.map_or(doc, |m: u128| m.max(doc)));
            }

            // If no iterators had any doc_id, or there are no iterators, return None
            // (This handles the case of an empty AndIter)
            let max_doc = max_doc?;

            // Advance all iterators to at least max_doc
            let mut all_equal = true;
            for iter in self.iters.iter_mut() {
                iter.skip_to(max_doc);
                let after = iter.doc_id();
                if after != Some(max_doc) {
                    all_equal = false;
                }
            }
            if all_equal {
                for iter in self.iters.iter_mut() {
                    iter.next();
                }
                return Some(max_doc);
            }
            // Otherwise, loop again to find the next intersection
        }
    }

    /// Advances all child iterators to at least the given doc_id.
    ///
    /// After calling, all children will be positioned at or after doc_id, or exhausted.
    fn skip_to(&mut self, doc_id: u128) {
        for iter in &mut self.iters {
            iter.skip_to(doc_id);
        }
    }

    /// Returns the current document ID if all child iterators are positioned at the same doc_id, otherwise None.
    fn doc_id(&self) -> Option<u128> {
        // Return the current doc_id if all iterators are at the same doc_id
        let first_doc_id = self.iters.first()?.doc_id()?;
        if self
            .iters
            .iter()
            .all(|it| it.doc_id() == Some(first_doc_id))
        {
            Some(first_doc_id)
        } else {
            None
        }
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
