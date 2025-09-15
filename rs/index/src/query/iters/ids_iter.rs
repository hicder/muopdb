use crate::query::iters::{InvertedIndexIter, IterState};

/// `IdsIter` is an iterator over a sorted list of unique document IDs.
///
/// It supports sequential access and efficient skipping to a target ID.
/// Used as a building block for query processing.
///
/// - `next()` yields the next ID in order.
/// - `skip_to(doc_id)` advances to the first ID >= `doc_id`.
/// - `doc_id()` returns the current ID, or None if exhausted.
///
/// Example:
/// ```
/// use index::query::iters::{ids_iter::IdsIter, InvertedIndexIter};
/// let mut iter = IdsIter::new(vec![1, 3, 5]);
/// assert_eq!(iter.next(), Some(1));
/// iter.skip_to(4);
/// assert_eq!(iter.doc_id(), Some(5));
/// ```
/// This iterator is used for simple ID-based filters and as a base for more complex iterators.
#[derive(Debug, Clone)]
pub struct IdsIter {
    ids: Vec<u128>,
    state: IterState<usize>, // Tracks the current index in ids
}

impl IdsIter {
    pub fn new(mut ids: Vec<u128>) -> Self {
        // Ensure the IDs are sorted and unique
        ids.sort_unstable();
        ids.dedup();
        Self {
            ids,
            state: IterState::NotStarted,
        }
    }
}

impl InvertedIndexIter for IdsIter {
    /// Advances the iterator and returns the next document ID, or None if exhausted.
    ///
    /// Updates the current_doc_id to the next value, or None if at the end.
    ///
    /// # Example
    /// ```
    /// use index::query::iters::{ids_iter::IdsIter, InvertedIndexIter};
    /// let mut iter = IdsIter::new(vec![1, 3, 5]);
    /// assert_eq!(iter.next(), Some(1));
    /// assert_eq!(iter.next(), Some(3));
    /// assert_eq!(iter.next(), Some(5));
    /// assert_eq!(iter.next(), None);
    /// ```
    fn next(&mut self) -> Option<u128> {
        match self.state {
            IterState::NotStarted => {
                match self.ids.first() {
                    Some(&first_doc_id) => {
                        self.state = IterState::At(0);
                        Some(first_doc_id)
                    }
                    None => {
                        // No IDs to iterate -> exhausted
                        self.state = IterState::Exhausted;
                        None
                    }
                }
            }
            IterState::At(index) => {
                if index + 1 < self.ids.len() {
                    // Move to the next ID
                    let next_index = index + 1;
                    let doc_id = self.ids[next_index];
                    self.state = IterState::At(next_index);
                    Some(doc_id)
                } else {
                    // Reached the end -> exhausted
                    self.state = IterState::Exhausted;
                    None
                }
            }
            IterState::Exhausted => None,
        }
    }

    /// Advances the iterator to the first document ID that is greater than or equal to `doc_id`.
    ///
    /// Updates current_doc_id to the found value, or None if past the end.
    ///
    /// # Example
    /// ```
    /// use index::query::iters::{ids_iter::IdsIter, InvertedIndexIter};
    /// let mut iter = IdsIter::new(vec![1, 3, 5]);
    /// iter.skip_to(4);
    /// assert_eq!(iter.doc_id(), Some(5));
    /// iter.skip_to(10);
    /// assert_eq!(iter.doc_id(), None);
    /// ```
    fn skip_to(&mut self, doc_id: u128) {
        let start = match self.state {
            IterState::NotStarted => 0,
            IterState::At(index) => index,
            IterState::Exhausted => return,
        };

        match self.ids[start..].binary_search(&doc_id) {
            Ok(i) => {
                let idx = start + i;
                self.state = IterState::At(idx);
            }
            Err(i) => {
                let idx = start + i;
                if idx < self.ids.len() {
                    self.state = IterState::At(idx);
                } else {
                    self.state = IterState::Exhausted;
                }
            }
        }
    }

    /// Returns the current document ID, or None if the iterator is exhausted or not started.
    fn doc_id(&self) -> Option<u128> {
        match self.state {
            IterState::At(index) => Some(self.ids[index]),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::iters::IdsIter;

    #[test]
    fn test_ids_iter_basic() {
        let ids = vec![1, 3, 5, 7, 9];
        let mut iter = IdsIter::new(ids);

        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.doc_id(), Some(1));

        assert_eq!(iter.next(), Some(3));
        assert_eq!(iter.doc_id(), Some(3));

        assert_eq!(iter.next(), Some(5));
        assert_eq!(iter.doc_id(), Some(5));

        assert_eq!(iter.next(), Some(7));
        assert_eq!(iter.doc_id(), Some(7));

        assert_eq!(iter.next(), Some(9));
        assert_eq!(iter.doc_id(), Some(9));

        assert_eq!(iter.next(), None);
        assert_eq!(iter.doc_id(), None);
    }

    #[test]
    fn test_ids_iter_skip_to() {
        let ids = vec![1, 3, 5, 7, 9];
        let mut iter = IdsIter::new(ids);

        // Skip to 5
        iter.skip_to(5);
        assert_eq!(iter.doc_id(), Some(5));
        assert_eq!(iter.next(), Some(7));

        // Skip to 8 (should land on 9)
        iter.skip_to(8);
        assert_eq!(iter.doc_id(), Some(9));
        assert_eq!(iter.next(), None);

        // Skip to 10 (should be None)
        iter.skip_to(10);
        assert_eq!(iter.doc_id(), None);
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_ids_iter_skip_to_exact_match() {
        let ids = vec![1, 3, 5, 7, 9];
        let mut iter = IdsIter::new(ids);

        // Skip to exact match
        iter.skip_to(3);
        assert_eq!(iter.doc_id(), Some(3));
        assert_eq!(iter.next(), Some(5));
    }

    #[test]
    fn test_ids_iter_empty() {
        let ids = vec![];
        let mut iter = IdsIter::new(ids);

        assert_eq!(iter.next(), None);
        assert_eq!(iter.doc_id(), None);

        iter.skip_to(5);
        assert_eq!(iter.doc_id(), None);
    }

    #[test]
    fn test_ids_iter_skip_to_before_first() {
        let ids = vec![5, 7, 9];
        let mut iter = IdsIter::new(ids);

        // Skip to value before first ID
        iter.skip_to(2);
        assert_eq!(iter.doc_id(), Some(5));
        assert_eq!(iter.next(), Some(7));
    }

    #[test]
    fn test_ids_iter_large_values() {
        let ids = vec![u128::MAX - 10, u128::MAX - 5, u128::MAX];
        let mut iter = IdsIter::new(ids);

        assert_eq!(iter.next(), Some(u128::MAX - 10));
        assert_eq!(iter.doc_id(), Some(u128::MAX - 10));

        assert_eq!(iter.next(), Some(u128::MAX - 5));
        assert_eq!(iter.doc_id(), Some(u128::MAX - 5));

        assert_eq!(iter.next(), Some(u128::MAX));
        assert_eq!(iter.doc_id(), Some(u128::MAX));

        assert_eq!(iter.next(), None);
        assert_eq!(iter.doc_id(), None);
    }
}
