use std::sync::Arc;

use quantization::quantization::Quantizer;

use super::index::Spann;

/// An iterator over valid documents in the SPANN index.
pub struct SpannIter<Q: Quantizer> {
    index: Arc<Spann<Q>>,
    next_point_id: u32,
}

impl<Q: Quantizer> SpannIter<Q> {
    /// Creates a new `SpannIter` for the given index.
    ///
    /// # Arguments
    /// * `index` - The SPANN index to iterate over.
    ///
    /// # Returns
    /// * `Self` - A new iterator instance.
    pub fn new(index: Arc<Spann<Q>>) -> Self {
        Self {
            index,
            next_point_id: 0,
        }
    }

    /// Returns the next valid document ID and its vector data.
    ///
    /// # Returns
    /// * `Option<(u128, &[Q::QuantizedT])>` - The next valid document ID and its quantized vector data, or `None` if the end is reached.
    pub async fn next(&mut self) -> Option<(u128, &[Q::QuantizedT])> {
        loop {
            if let Some(doc_id) = self.index.get_doc_id(self.next_point_id).await {
                if !self.index.is_invalidated(doc_id).await {
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
