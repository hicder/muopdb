use std::sync::Arc;

use parking_lot::Mutex;
use quantization::quantization::Quantizer;

use super::index::Spann;
use crate::utils::SearchContext;

pub struct SpannIter<Q: Quantizer> {
    index: Arc<Spann<Q>>,
    next_point_id: u32,
    search_context: Arc<Mutex<SearchContext>>,
}

impl<Q: Quantizer> SpannIter<Q> {
    pub fn new(index: Arc<Spann<Q>>) -> Self {
        Self {
            index,
            next_point_id: 0,
            search_context: Arc::new(Mutex::new(SearchContext::new(false))),
        }
    }

    pub fn next(&mut self) -> Option<(u128, &[Q::QuantizedT])> {
        if let Some(doc_id) = self.index.get_doc_id(self.next_point_id) {
            let vector = self
                .index
                .get_vector(self.next_point_id, self.search_context.clone())
                .unwrap();
            self.next_point_id += 1;
            Some((doc_id, vector))
        } else {
            None
        }
    }
}
