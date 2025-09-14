use crate::query::iter::InvertedIndexIter;

#[allow(unused)]
pub struct OrIter {
    iters: Vec<Box<dyn InvertedIndexIter>>,
}

impl OrIter {
    pub fn new(iters: Vec<Box<dyn InvertedIndexIter>>) -> Self {
        Self { iters }
    }
}

impl InvertedIndexIter for OrIter {
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
