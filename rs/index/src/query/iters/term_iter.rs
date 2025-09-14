use compression::elias_fano::ef::EliasFanoDecodingIterator;

use crate::query::iter::InvertedIndexIter;

pub struct TermIter<'a> {
    ef_iter: EliasFanoDecodingIterator<'a>,
    doc_id: Option<u64>,
}

impl<'a> TermIter<'a> {
    pub fn new(ef_iter: EliasFanoDecodingIterator<'a>) -> Self {
        Self {
            ef_iter,
            doc_id: None,
        }
    }
}

impl<'a> InvertedIndexIter for TermIter<'a> {
    fn next(&mut self) -> Option<u64> {
        self.doc_id = self.ef_iter.next();
        self.doc_id
    }

    fn skip_to(&mut self, doc_id: u64) {
        // TODO(hicder): Properly implement skip_to under EF iterator
        while self.doc_id().unwrap_or(u64::MAX) < doc_id {
            self.next();
        }
    }

    fn doc_id(&self) -> Option<u64> {
        self.doc_id
    }
}
