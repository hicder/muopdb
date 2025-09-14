use anyhow::Result;
use proto::muopdb::DocumentFilter;

use crate::query::iter::InvertedIndexIter;

#[allow(unused)]
pub struct Planner {
    query: DocumentFilter,
}

impl Planner {
    pub fn new(query: DocumentFilter) -> Self {
        Self { query }
    }

    pub fn plan(&self) -> Result<Box<dyn InvertedIndexIter>> {
        todo!()
    }
}
