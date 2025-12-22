use anyhow::Result;

use crate::query::iters::{InvertedIndexIter, IterState};
use crate::terms::index::TermIndex;

pub struct TermIter {
    point_ids: Vec<u32>,
    state: IterState<usize>,
}

impl TermIter {
    pub fn new(term_index: &TermIndex, term_id: u64) -> Result<Self> {
        let point_ids_iter = term_index.get_posting_list_iterator(term_id)?;

        Ok(Self {
            point_ids: point_ids_iter.collect(),
            state: IterState::NotStarted,
        })
    }
}

impl InvertedIndexIter for TermIter {
    fn next(&mut self) -> Option<u32> {
        match self.state {
            IterState::NotStarted => match self.point_ids.first() {
                Some(&point_id) => {
                    self.state = IterState::At(0);
                    Some(point_id)
                }
                None => {
                    self.state = IterState::Exhausted;
                    None
                }
            },
            IterState::At(index) => {
                if index + 1 < self.point_ids.len() {
                    let next_index = index + 1;
                    let point_id = self.point_ids[next_index];
                    self.state = IterState::At(next_index);
                    Some(point_id)
                } else {
                    self.state = IterState::Exhausted;
                    None
                }
            }
            IterState::Exhausted => None,
        }
    }

    fn skip_to(&mut self, point_id: u32) {
        let start = match self.state {
            IterState::NotStarted => 0,
            IterState::At(index) => index,
            IterState::Exhausted => return,
        };

        match self.point_ids[start..].binary_search(&point_id) {
            Ok(i) => {
                let idx = start + i;
                self.state = IterState::At(idx);
            }
            Err(i) => {
                let idx = start + i;
                if idx < self.point_ids.len() {
                    self.state = IterState::At(idx);
                } else {
                    self.state = IterState::Exhausted;
                }
            }
        }
    }

    fn point_id(&self) -> Option<u32> {
        match self.state {
            IterState::At(index) => Some(self.point_ids[index]),
            _ => None,
        }
    }
}
