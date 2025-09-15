use crate::query::iters::{InvertedIndexIter, Iter, IterState};

/// `AndIter` yields the intersection of all its child iterators.
///
/// It only yields document IDs that are present in **every** child iterator.
///
/// - If any child iterator is exhausted, the AND iterator is exhausted.
/// - Advances all children in lockstep to find common IDs.
///
/// Example:
/// ```
/// use index::query::iters::{and_iter::AndIter, InvertedIndexIter, Iter};
/// use index::query::iters::ids_iter::IdsIter;
/// let a = Iter::Ids(IdsIter::new(vec![1, 3, 5]));
/// let b = Iter::Ids(IdsIter::new(vec![3, 5, 7]));
/// let c = Iter::Ids(IdsIter::new(vec![3, 4, 5]));
/// let mut and_iter = AndIter::new(vec![a, b, c]);
/// let mut results = Vec::new();
/// while let Some(doc) = and_iter.next() {
///     results.push(doc);
/// }
/// assert_eq!(results, vec![3, 5]);
/// ```
/// Used to answer queries like "find documents that match all these conditions".
pub struct AndIter<'a> {
    iters: Vec<Iter<'a>>,
    state: IterState<u128>, // Current doc_id
}

impl<'a> AndIter<'a> {
    pub fn new(iters: Vec<Iter<'a>>) -> Self {
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

    /// Align all children to the same doc_id, returning that doc_id if successful.
    /// If any child is exhausted during alignment, returns None.
    fn align(&mut self) -> Option<u128> {
        loop {
            let mut max_doc = None;

            // Gather doc_ids, fail fast if any exhausted
            for child in self.iters.iter() {
                match child.doc_id() {
                    Some(doc) => {
                        max_doc = Some(max_doc.map_or(doc, |m: u128| m.max(doc)));
                    }
                    None => return None, // uninitialized or exhausted -> no intersection
                }
            }

            let target = max_doc?;

            // Try to align all children to `target`
            let mut all_equal = true;
            for child in self.iters.iter_mut() {
                child.skip_to(target);
                match child.doc_id() {
                    Some(doc) if doc == target => {}
                    Some(_) => {
                        all_equal = false;
                    }
                    None => return None, // child exhausted
                }
            }

            if all_equal {
                return Some(target);
            }
            // else loop until convergence
        }
    }
}

impl<'a> InvertedIndexIter for AndIter<'a> {
    /// Advances the iterator and returns the next document ID, or None if exhausted.
    fn next(&mut self) -> Option<u128> {
        match self.state {
            IterState::NotStarted => {
                // Initialize children
                for child in self.iters.iter_mut() {
                    if child.next().is_none() {
                        self.state = IterState::Exhausted;
                        return None;
                    }
                }
                match self.align() {
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
            IterState::At(_) => {
                // Advance first child (arbitrary choice)
                if self.iters[0].next().is_none() {
                    self.state = IterState::Exhausted;
                    return None;
                }
                match self.align() {
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
            IterState::Exhausted => None,
        }
    }

    /// Advances all child iterators to at least the target document ID, setting the state to the new aligned doc ID if found, or exhausted if not.
    /// If any child is exhausted, the AND iterator becomes exhausted.
    fn skip_to(&mut self, target: u128) {
        match self.state {
            IterState::Exhausted => {}
            _ => {
                for child in self.iters.iter_mut() {
                    child.skip_to(target);
                    if child.doc_id().is_none() {
                        self.state = IterState::Exhausted;
                        return;
                    }
                }
                match self.align() {
                    Some(doc) => self.state = IterState::At(doc),
                    None => self.state = IterState::Exhausted,
                }
            }
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

    #[test]
    fn test_and_iter_no_intersection() {
        // No intersection
        let a = ids(&[1, 2, 3]);
        let b = ids(&[4, 5, 6]);
        let mut iter = AndIter::new(vec![a, b]);
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_and_iter_superset() {
        // One child is a superset of another
        let a = ids(&[1, 2, 3, 4, 5]);
        let b = ids(&[2, 4]);
        let mut iter = AndIter::new(vec![a, b]);
        let mut results = Vec::new();
        while let Some(doc) = iter.next() {
            results.push(doc);
        }
        assert_eq!(results, vec![2, 4]);
    }

    #[test]
    fn test_and_iter_all_empty_children() {
        let mut iter = AndIter::new(vec![ids(&[]), ids(&[])]);
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_and_with_empty_child() {
        // AND with one child empty: should yield nothing
        let a = ids(&[1, 2, 3]);
        let b = ids(&[]);
        let mut iter = AndIter::new(vec![a, b]);
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_empty_and() {
        // AND of empty: should yield nothing
        let mut iter = AndIter::new(vec![]);
        assert_eq!(iter.next(), None);
    }
}
