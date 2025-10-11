use std::sync::Arc;

use anyhow::Result;
use compression::elias_fano::ef::EliasFanoDecodingIterator;

use crate::query::iters::InvertedIndexIter;
use crate::terms::index::TermIndex;

pub struct TermIter<'a> {
    ef_iter: EliasFanoDecodingIterator<'a>,
    doc_id: Option<u128>,
}

impl<'a> TermIter<'a> {
    pub fn new(ef_iter: EliasFanoDecodingIterator<'a>) -> Self {
        Self {
            ef_iter,
            doc_id: None,
        }
    }
}

pub struct ArcTermIter {
    doc_ids: Vec<u128>,
    current_index: usize,
}

impl ArcTermIter {
    pub fn new(term_index: Arc<TermIndex>, term_id: u64) -> Result<Self> {
        let ef_iter = term_index.get_posting_list_iterator(term_id)?;
        let doc_ids: Vec<u128> = ef_iter.map(|v| v as u128).collect();

        Ok(Self {
            doc_ids,
            current_index: 0,
        })
    }
}

impl InvertedIndexIter for ArcTermIter {
    fn next(&mut self) -> Option<u128> {
        if self.current_index < self.doc_ids.len() {
            let doc_id = self.doc_ids[self.current_index];
            self.current_index += 1;
            Some(doc_id)
        } else {
            None
        }
    }

    fn skip_to(&mut self, doc_id: u128) {
        // Skip forward until we find a document >= doc_id
        while self.current_index < self.doc_ids.len() && self.doc_ids[self.current_index] < doc_id {
            self.current_index += 1;
        }
    }

    fn doc_id(&self) -> Option<u128> {
        if self.current_index < self.doc_ids.len() {
            Some(self.doc_ids[self.current_index])
        } else {
            None
        }
    }
}

impl<'a> InvertedIndexIter for TermIter<'a> {
    fn next(&mut self) -> Option<u128> {
        self.doc_id = self.ef_iter.next().map(|v| v as u128);
        self.doc_id
    }

    fn skip_to(&mut self, doc_id: u128) {
        // TODO(hicder): Properly implement skip_to under EF iterator
        while self.doc_id().unwrap_or(u128::MAX) < doc_id {
            self.next();
        }
    }

    fn doc_id(&self) -> Option<u128> {
        self.doc_id
    }
}
