use anyhow::Result;
use compression::elias_fano::ef::EliasFanoDecodingIterator;

use crate::query::iters::{InvertedIndexIter, IterState};
use crate::terms::index::TermIndex;

pub struct TermIter<'a> {
    iter: EliasFanoDecodingIterator<'a, u32>,
    state: IterState<u32>,
}

impl<'a> TermIter<'a> {
    pub fn new(term_index: &'a TermIndex, term_id: u64) -> Result<Self> {
        let iter = term_index.get_posting_list_iterator(term_id)?;
        Ok(Self {
            iter,
            state: IterState::NotStarted,
        })
    }
}

impl<'a> InvertedIndexIter for TermIter<'a> {
    fn next(&mut self) -> Option<u32> {
        match self.state {
            IterState::NotStarted => {
                let point = self.iter.current();
                self.state = if point.is_some() {
                    IterState::At(point.unwrap())
                } else {
                    IterState::Exhausted
                };
                point
            }
            IterState::At(_) => {
                let point = self.iter.next();
                self.state = if point.is_some() {
                    IterState::At(point.unwrap())
                } else {
                    IterState::Exhausted
                };
                point
            }
            IterState::Exhausted => None,
        }
    }

    fn skip_to(&mut self, point_id: u32) {
        self.iter.skip_to(point_id);
        let point = self.iter.current();
        self.state = if point.is_some() {
            IterState::At(point.unwrap())
        } else {
            IterState::Exhausted
        };
    }

    fn point_id(&mut self) -> Option<u32> {
        match self.state {
            IterState::At(point) => Some(point),
            _ => None,
        }
    }
}
