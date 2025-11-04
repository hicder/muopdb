use anyhow::Result;

use crate::query::iters::InvertedIndexIter;
use crate::terms::index::TermIndex;

pub struct TermIter {
    point_ids: Vec<u32>,
    current_index: usize,
}

impl TermIter {
    pub fn new(term_index: &TermIndex, term_id: u64) -> Result<Self> {
        // Get posting list iterator (yields point_ids)
        let point_ids_iter = term_index.get_posting_list_iterator(term_id)?;

        Ok(Self {
            point_ids: point_ids_iter.collect(),
            current_index: 0,
        })
    }
}

impl InvertedIndexIter for TermIter {
    fn next(&mut self) -> Option<u32> {
        if self.current_index < self.point_ids.len() {
            let point_id = self.point_ids[self.current_index];
            self.current_index += 1;
            Some(point_id)
        } else {
            None
        }
    }

    fn skip_to(&mut self, point_id: u32) {
        // Skip forward until we find a document >= point_id
        while self.current_index < self.point_ids.len()
            && self.point_ids[self.current_index] < point_id
        {
            self.current_index += 1;
        }
    }

    fn point_id(&self) -> Option<u32> {
        if self.current_index < self.point_ids.len() {
            Some(self.point_ids[self.current_index])
        } else {
            None
        }
    }
}
