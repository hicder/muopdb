pub trait InvertedIndexIter {
    fn next(&mut self) -> Option<u128>;

    fn skip_to(&mut self, doc_id: u128);

    fn doc_id(&self) -> Option<u128>;
}
