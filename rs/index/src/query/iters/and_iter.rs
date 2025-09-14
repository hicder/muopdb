use crate::query::iter::{InvertedIndexIter, Iter};

pub struct AndIter<'a> {
    pub iters: Vec<Iter<'a>>,
}

impl<'a> AndIter<'a> {
    pub fn new(iters: Vec<Iter<'a>>) -> Self {
        Self { iters }
    }
}

impl<'a> InvertedIndexIter for AndIter<'a> {
    fn next(&mut self) -> Option<u128> {
        todo!()
    }

    fn skip_to(&mut self, _doc_id: u128) {
        todo!()
    }

    fn doc_id(&self) -> Option<u128> {
        todo!()
    }
}
