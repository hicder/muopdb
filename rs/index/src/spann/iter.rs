use std::sync::Arc;

use quantization::quantization::Quantizer;

use super::index::Spann;

pub struct SpannIter<Q: Quantizer> {
    index: Arc<Spann<Q>>,
    next_point_id: u32,
}

impl<Q: Quantizer> SpannIter<Q> {
    pub fn new(index: Arc<Spann<Q>>) -> Self {
        Self {
            index,
            next_point_id: 0,
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Option<(u128, &[Q::QuantizedT])> {
        loop {
            if let Some(doc_id) = self.index.get_doc_id(self.next_point_id) {
                if !self.index.is_invalidated(doc_id) {
                    let vector = self.index.get_vector(self.next_point_id).unwrap();
                    self.next_point_id += 1;
                    return Some((doc_id, vector));
                }
            } else {
                return None;
            }
            self.next_point_id += 1;
        }
    }
}
