use anyhow::Result;
use quantization::quantization::Quantizer;

use crate::query::iters::InvertedIndexIter;
use crate::spann::index::Spann;
use crate::terms::index::TermIndex;

pub struct TermIter {
    doc_ids: Vec<u128>,
    current_index: usize,
}

impl TermIter {
    pub fn new<Q: Quantizer>(
        term_index: &TermIndex,
        spann_index: &Spann<Q>,
        term_id: u64,
    ) -> Result<Self> {
        // Get posting list iterator (yields point_ids)
        let point_ids_iter = term_index.get_posting_list_iterator(term_id)?;

        // Map point_ids to doc_ids using the Spann index
        let mut doc_ids = point_ids_iter
            .filter_map(|point_id| spann_index.get_doc_id(point_id))
            .collect::<Vec<u128>>();

        // Ensure the doc_ids are sorted and unique
        doc_ids.sort_unstable();
        doc_ids.dedup();
        Ok(Self {
            doc_ids,
            current_index: 0,
        })
    }
}

impl InvertedIndexIter for TermIter {
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
