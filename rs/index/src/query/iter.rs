pub trait InvertedIndexIter {
    fn next(&mut self) -> Option<u64>;

    fn skip_to(&mut self, doc_id: u64);

    fn doc_id(&self) -> Option<u64>;
}
