use crate::query::iters::and_iter::AndIter;
use crate::query::iters::ids_iter::IdsIter;
use crate::query::iters::or_iter::OrIter;
use crate::query::iters::term_iter::TermIter;

pub enum Iter<'a> {
    And(AndIter<'a>),
    Or(OrIter<'a>),
    Ids(IdsIter),
    Term(TermIter<'a>),
}

impl<'a> InvertedIndexIter for Iter<'a> {
    fn next(&mut self) -> Option<u128> {
        match self {
            Iter::And(iter) => iter.next(),
            Iter::Or(iter) => iter.next(),
            Iter::Ids(iter) => iter.next(),
            Iter::Term(iter) => iter.next(),
        }
    }

    fn skip_to(&mut self, doc_id: u128) {
        match self {
            Iter::And(iter) => iter.skip_to(doc_id),
            Iter::Or(iter) => iter.skip_to(doc_id),
            Iter::Ids(iter) => iter.skip_to(doc_id),
            Iter::Term(iter) => iter.skip_to(doc_id),
        }
    }

    fn doc_id(&self) -> Option<u128> {
        match self {
            Iter::And(iter) => iter.doc_id(),
            Iter::Or(iter) => iter.doc_id(),
            Iter::Ids(iter) => iter.doc_id(),
            Iter::Term(iter) => iter.doc_id(),
        }
    }
}

pub trait InvertedIndexIter {
    fn next(&mut self) -> Option<u128>;

    fn skip_to(&mut self, doc_id: u128);

    fn doc_id(&self) -> Option<u128>;
}
