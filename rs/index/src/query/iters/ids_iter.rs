use crate::query::iters::InvertedIndexIter;

pub struct IdsIter {
    ids: Vec<u128>,
    current_index: usize,
    current_doc_id: Option<u128>,
}

impl IdsIter {
    pub fn new(ids: Vec<u128>) -> Self {
        Self {
            ids,
            current_index: 0,
            current_doc_id: None,
        }
    }
}

impl InvertedIndexIter for IdsIter {
    fn next(&mut self) -> Option<u128> {
        if self.current_index < self.ids.len() {
            self.current_doc_id = Some(self.ids[self.current_index]);
            self.current_index += 1;
            self.current_doc_id
        } else {
            self.current_doc_id = None;
            None
        }
    }

    fn skip_to(&mut self, doc_id: u128) {
        // Find the first doc_id >= target doc_id
        while self.current_index < self.ids.len() {
            let current_id = self.ids[self.current_index];
            if current_id >= doc_id {
                self.current_doc_id = Some(current_id);
                break;
            }
            self.current_index += 1;
        }

        // If we've gone past all IDs, set current_doc_id to None
        if self.current_index >= self.ids.len() {
            self.current_doc_id = None;
        }
    }

    fn doc_id(&self) -> Option<u128> {
        self.current_doc_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(iter.next(), Some(5));

        // Skip to 8 (should land on 9)
        iter.skip_to(8);
        assert_eq!(iter.doc_id(), Some(9));
        assert_eq!(iter.next(), Some(9));

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
        assert_eq!(iter.next(), Some(3));
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
        assert_eq!(iter.next(), Some(5));
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
