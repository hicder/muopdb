use std::collections::HashMap;

use anyhow::{anyhow, Result};
use utils::on_disk_ordered_map::builder::OnDiskOrderedMapBuilder;

/// Single user term builder.
pub struct TermBuilder {
    /// Map from term string to term ID.
    pub term_map: OnDiskOrderedMapBuilder,
    next_term_id: u64,
    /// In-memory posting lists for each term ID. Each posting list is a list of point ID (u32).
    posting_lists: HashMap<u64, Vec<u32>>,

    built: bool,
}

impl TermBuilder {
    pub fn new() -> Result<Self> {
        Ok(Self {
            term_map: OnDiskOrderedMapBuilder::new(),
            next_term_id: 0,
            posting_lists: HashMap::new(),
            built: false,
        })
    }

    pub fn add(&mut self, point_id: u32, key: String) -> Result<()> {
        if self.built {
            return Err(anyhow!("TermBuilder is already built"));
        }

        let term_id = self.persist_and_get_term_id(key);
        self.posting_lists
            .entry(term_id)
            .or_default()
            .push(point_id);
        Ok(())
    }

    /// Adds a term to the map and returns its ID.
    /// If the term already exists, returns its ID.
    /// Otherwise, assigns a new ID and returns it.
    fn persist_and_get_term_id(&mut self, key: String) -> u64 {
        let term_id = self.term_map.add_or_get(key, self.next_term_id);
        if term_id == self.next_term_id {
            self.next_term_id += 1;
        }
        term_id
    }

    pub fn build(&mut self) -> Result<()> {
        if self.built {
            return Err(anyhow!("TermBuilder is already built"));
        }
        self.built = true;

        // Sort the posting lists so we can use Elias Fano encoding.
        for (_, posting_list) in self.posting_lists.iter_mut() {
            posting_list.sort();
            posting_list.dedup();
        }
        Ok(())
    }

    pub fn is_built(&self) -> bool {
        self.built
    }

    pub fn num_terms(&self) -> u64 {
        self.next_term_id
    }

    pub fn get_posting_list(&self, term: &str) -> Option<&[u32]> {
        if let Some(term_id) = self.term_map.get_value(term) {
            return self.get_posting_list_by_id(term_id);
        }
        None
    }

    pub fn get_posting_list_by_id(&self, term_id: u64) -> Option<&[u32]> {
        self.posting_lists.get(&term_id).map(|v| v.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_term_builder() {
        let mut builder = TermBuilder::new().unwrap();
        builder.add(0, "a".to_string()).unwrap();
        builder.add(0, "c".to_string()).unwrap();
        builder.add(1, "b".to_string()).unwrap();
        builder.add(2, "c".to_string()).unwrap();

        assert_eq!(*builder.get_posting_list("a").unwrap(), vec![0]);
        assert_eq!(*builder.get_posting_list("b").unwrap(), vec![1]);
        assert_eq!(*builder.get_posting_list("c").unwrap(), vec![0, 2]);
    }
}
