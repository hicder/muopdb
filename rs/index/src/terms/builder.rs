use std::collections::HashMap;
use std::path::Path;

use anyhow::{anyhow, Result};
use utils::on_disk_ordered_map::builder::OnDiskOrderedMapBuilder;

use super::scratch::Scratch;

pub struct TermBuilder {
    pub term_map: OnDiskOrderedMapBuilder,
    next_term_id: u64,
    scratch_file: Scratch,
    posting_lists: HashMap<u64, Vec<u64>>,

    built: bool,
}

impl TermBuilder {
    pub fn new(dir: &str) -> Self {
        let dir_path = Path::new(dir);
        let scratch_file_path = dir_path.join("scratch.tmp");
        Self {
            term_map: OnDiskOrderedMapBuilder::new(),
            next_term_id: 0,
            scratch_file: Scratch::new(scratch_file_path.as_path().to_str().unwrap()),
            posting_lists: HashMap::new(),
            built: false,
        }
    }

    pub fn add(&mut self, doc_id: u64, key: String) -> Result<()> {
        if self.built {
            return Err(anyhow!("TermBuilder is already built"));
        }
        #[cfg(debug_assertions)]
        {
            log::debug!("Adding doc: {}, term: {}", doc_id, key);
        }

        let term_id = self.persist_and_get_term_id(key);
        self.scratch_file.write(doc_id, term_id)?;
        self.posting_lists
            .entry(term_id)
            .or_insert_with(Vec::new)
            .push(doc_id);
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

    pub fn get_posting_list(&self, term: &str) -> Option<&Vec<u64>> {
        if let Some(term_id) = self.term_map.get_value(term) {
            return self.get_posting_list_by_id(term_id);
        }
        None
    }

    pub fn get_posting_list_by_id(&self, term_id: u64) -> Option<&Vec<u64>> {
        self.posting_lists.get(&term_id)
    }
}

#[cfg(test)]
mod tests {
    use tempdir::TempDir;

    use super::*;

    #[test]
    fn test_term_builder() {
        let tmp_dir = TempDir::new("test_term_builder").unwrap();
        let base_directory = tmp_dir.path().to_str().unwrap();

        let mut builder = TermBuilder::new(base_directory);
        builder.add(0, "a".to_string()).unwrap();
        builder.add(0, "c".to_string()).unwrap();
        builder.add(1, "b".to_string()).unwrap();
        builder.add(2, "c".to_string()).unwrap();

        assert_eq!(*builder.get_posting_list("a").unwrap(), vec![0]);
        assert_eq!(*builder.get_posting_list("b").unwrap(), vec![1]);
        assert_eq!(*builder.get_posting_list("c").unwrap(), vec![0, 2]);
    }
}
