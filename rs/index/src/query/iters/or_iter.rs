use crate::query::iters::{InvertedIndexIter, Iter, IterState};

/// `OrIter` yields the union of all its child iterators.
///
/// It yields every document ID that appears in any child iterator, without duplicates, in sorted order.
///
/// - At each step, finds the smallest current doc_id among all children, yields it, and advances all children at that doc_id.
/// - If all children are exhausted, the OR iterator is exhausted.
///
/// Example:
/// ```
/// use index::query::iters::{or_iter::OrIter, InvertedIndexIter, Iter};
/// use index::query::iters::ids_iter::IdsIter;
/// let a = Iter::Ids(IdsIter::new(vec![1, 3, 5]));
/// let b = Iter::Ids(IdsIter::new(vec![3, 5, 7]));
/// let c = Iter::Ids(IdsIter::new(vec![2, 3, 5]));
/// let mut or_iter = OrIter::new(vec![a, b, c]);
/// let mut results = Vec::new();
/// while let Some(doc) = or_iter.next() {
///     results.push(doc);
/// }
/// assert_eq!(results, vec![1, 2, 3, 5, 7]);
/// ```
/// Used to answer queries like "find documents that match any of these conditions".
pub struct OrIter {
    iters: Vec<Iter>,
    state: IterState<u128>, // Current doc_id
}

impl OrIter {
    pub fn new(iters: Vec<Iter>) -> Self {
        if iters.is_empty() {
            // Set to exhausted immediately if no children
            return Self {
                iters,
                state: IterState::Exhausted,
            };
        }
        Self {
            iters,
            state: IterState::NotStarted,
        }
    }

    /// Find the smallest doc_id across all children, or None if all exhausted.
    fn min_doc(&self) -> Option<u128> {
        self.iters.iter().filter_map(|c| c.doc_id()).min()
    }

    /// Advance children that are exactly at `doc`, so they will compete for the next min.
    fn advance_consumed(&mut self, doc: u128) {
        for child in self.iters.iter_mut() {
            if child.doc_id() == Some(doc) {
                child.next(); // move this child forward once
            }
        }
    }
}

impl InvertedIndexIter for OrIter {
    /// Advances the iterator and returns the next document ID, or None if exhausted.
    fn next(&mut self) -> Option<u128> {
        match self.state {
            IterState::NotStarted => {
                // Initialize all children (prime them at first doc)
                for child in self.iters.iter_mut() {
                    child.next();
                }
            }
            IterState::At(prev_doc) => {
                // Consume children that contributed to last doc
                self.advance_consumed(prev_doc);
            }
            IterState::Exhausted => return None,
        }

        // Pick next min doc
        match self.min_doc() {
            Some(doc) => {
                self.state = IterState::At(doc);
                Some(doc)
            }
            None => {
                self.state = IterState::Exhausted;
                None
            }
        }
    }

    /// Advances all child iterators to at least the target document ID, setting the state to the new minimum doc ID found.
    /// If all children are exhausted, the OR iterator becomes exhausted.
    fn skip_to(&mut self, target: u128) {
        if matches!(self.state, IterState::Exhausted) {
            return;
        }

        for child in self.iters.iter_mut() {
            child.skip_to(target);
        }

        match self.min_doc() {
            Some(doc) => self.state = IterState::At(doc),
            None => self.state = IterState::Exhausted,
        }
    }

    /// Returns the current document ID, or None if the iterator is exhausted or not started.
    fn doc_id(&self) -> Option<u128> {
        match self.state {
            IterState::At(doc) => Some(doc),
            _ => None,
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

    #[test]
    fn test_or_iter_no_overlap() {
        // Children with no overlap
        let a = ids(&[1, 2, 3]);
        let b = ids(&[4, 5, 6]);
        let mut iter = OrIter::new(vec![a, b]);
        let mut results = Vec::new();
        while let Some(doc) = iter.next() {
            results.push(doc);
        }
        assert_eq!(results, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_or_iter_superset() {
        // One child is a superset of another
        let a = ids(&[1, 2, 3, 4, 5]);
        let b = ids(&[2, 4]);
        let mut iter = OrIter::new(vec![a, b]);
        let mut results = Vec::new();
        while let Some(doc) = iter.next() {
            results.push(doc);
        }
        assert_eq!(results, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_or_iter_all_empty_children() {
        let mut iter = OrIter::new(vec![ids(&[]), ids(&[])]);
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_or_iter_interleaved() {
        // Children with interleaved values
        let a = ids(&[1, 3, 5]);
        let b = ids(&[2, 4, 6]);
        let mut iter = OrIter::new(vec![a, b]);
        let mut results = Vec::new();
        while let Some(doc) = iter.next() {
            results.push(doc);
        }
        assert_eq!(results, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_or_with_empty_child() {
        // OR with one child empty: should yield the non-empty child
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
    fn test_empty_or() {
        // OR of empty: should yield nothing
        let mut iter = OrIter::new(vec![]);
        assert_eq!(iter.next(), None);
    }
}
