use std::collections::HashMap;
use std::path::Path;

use anyhow::{anyhow, Result};
use utils::on_disk_ordered_map::builder::OnDiskOrderedMapBuilder;

use super::scratch::Scratch;

/// Single user term builder.
pub struct TermBuilder {
    /// Map from term string to term ID.
    pub term_map: OnDiskOrderedMapBuilder,
    next_term_id: u64,
    scratch_file: Scratch,
    /// In-memory posting lists for each term ID. Each posting list is a list of point ID (u32).
    posting_lists: HashMap<u64, Vec<u32>>,

    built: bool,
}

impl TermBuilder {
    pub fn new(scratch_file_path: &Path) -> Result<Self> {
        Ok(Self {
            term_map: OnDiskOrderedMapBuilder::new(),
            next_term_id: 0,
            scratch_file: Scratch::new(scratch_file_path)?,
            posting_lists: HashMap::new(),
            built: false,
        })
    }

    pub fn add(&mut self, point_id: u32, key: String) -> Result<()> {
        if self.built {
            return Err(anyhow!("TermBuilder is already built"));
        }

        let term_id = self.persist_and_get_term_id(key);
        self.scratch_file.write(point_id, term_id)?;
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

/// Builer struct for multiple users, each with their own TermBuilder.
pub struct MultiTermBuilder {
    /// Map from user ID to its corresponding TermBuilder.
    inner_builders: HashMap<u128, TermBuilder>,
    /// Base directory for the term builders' scratch files.
    base_directory: String,
    /// Whether the builders have been built.
    is_built: bool,
}

impl MultiTermBuilder {
    pub fn new(base_directory: String) -> Self {
        Self {
            inner_builders: HashMap::new(),
            base_directory,
            is_built: false,
        }
    }

    pub fn builders_iter_mut(&mut self) -> impl Iterator<Item = (&u128, &mut TermBuilder)> {
        self.inner_builders.iter_mut()
    }

    pub fn add(&mut self, user_id: u128, point_id: u32, key: String) -> Result<()> {
        let builder = match self.inner_builders.entry(user_id) {
            // Use existing builder
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            // Create a new builder
            std::collections::hash_map::Entry::Vacant(entry) => {
                let scratch_file_path =
                    Path::new(&self.base_directory).join(format!("scratch_user_{}.tmp", user_id));
                let new_builder = TermBuilder::new(&scratch_file_path)?;
                // Insert and get mutable reference
                entry.insert(new_builder)
            }
        };
        builder.add(point_id, key)
    }

    pub fn build(&mut self) -> Result<()> {
        if self.is_built {
            return Err(anyhow!("MultiTermBuilder is already built"));
        }
        for (_, builder) in self.inner_builders.iter_mut() {
            builder.build()?;
        }
        self.is_built = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use tempdir::TempDir;

    use super::*;

    #[test]
    fn test_term_builder() {
        let tmp_dir = TempDir::new("test_term_builder").unwrap();
        let scratch_file_path = tmp_dir.path().join("scratch.tmp");

        let mut builder = TermBuilder::new(scratch_file_path.as_path()).unwrap();
        builder.add(0, "a".to_string()).unwrap();
        builder.add(0, "c".to_string()).unwrap();
        builder.add(1, "b".to_string()).unwrap();
        builder.add(2, "c".to_string()).unwrap();

        assert_eq!(*builder.get_posting_list("a").unwrap(), vec![0]);
        assert_eq!(*builder.get_posting_list("b").unwrap(), vec![1]);
        assert_eq!(*builder.get_posting_list("c").unwrap(), vec![0, 2]);
    }

    #[test]
    fn test_multi_term_builder() {
        let tmp_dir = TempDir::new("test_multi_term_builder").unwrap();
        let base_directory = tmp_dir.path().to_str().unwrap().to_string();

        let mut multi_builder = MultiTermBuilder::new(base_directory);

        // Add terms for different users
        let user1 = 123u128;
        let user2 = 456u128;

        // User 1: Add some terms
        multi_builder.add(user1, 0, "apple".to_string()).unwrap();
        multi_builder.add(user1, 1, "banana".to_string()).unwrap();
        multi_builder.add(user1, 2, "apple".to_string()).unwrap(); // Same term, different point

        // User 2: Add some terms (can have same terms as user1)
        multi_builder.add(user2, 0, "apple".to_string()).unwrap();
        multi_builder.add(user2, 1, "orange".to_string()).unwrap();
        multi_builder.add(user2, 2, "grape".to_string()).unwrap();

        // Build all user builders
        multi_builder.build().unwrap();

        // Test that we can iterate through builders
        let mut builder_count = 0;
        for (user_id, builder) in multi_builder.builders_iter_mut() {
            builder_count += 1;

            match *user_id {
                123 => {
                    // User 1 should have 2 terms (apple, banana)
                    assert_eq!(builder.num_terms(), 2);
                    assert_eq!(*builder.get_posting_list("apple").unwrap(), vec![0, 2]);
                    assert_eq!(*builder.get_posting_list("banana").unwrap(), vec![1]);
                }
                456 => {
                    // User 2 should have 3 terms (apple, orange, grape)
                    assert_eq!(builder.num_terms(), 3);
                    assert_eq!(*builder.get_posting_list("apple").unwrap(), vec![0]);
                    assert_eq!(*builder.get_posting_list("orange").unwrap(), vec![1]);
                    assert_eq!(*builder.get_posting_list("grape").unwrap(), vec![2]);
                }
                _ => panic!("Unexpected user_id: {}", user_id),
            }
        }

        // Should have exactly 2 users
        assert_eq!(builder_count, 2);
    }

    #[test]
    fn test_multi_term_builder_error_handling() {
        let tmp_dir = TempDir::new("test_multi_term_builder_error").unwrap();
        let base_directory = tmp_dir.path().to_str().unwrap().to_string();

        let mut multi_builder = MultiTermBuilder::new(base_directory);

        // Add some terms
        multi_builder.add(123u128, 0, "test".to_string()).unwrap();

        // Build once
        multi_builder.build().unwrap();

        // Try to build again - should fail
        let result = multi_builder.build();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("already built"));

        // Try to add after building - should fail at the individual builder level
        let result = multi_builder.add(123u128, 1, "new_term".to_string());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("already built"));
    }

    #[test]
    fn test_multi_term_builder_empty() {
        let tmp_dir = TempDir::new("test_multi_term_builder_empty").unwrap();
        let base_directory = tmp_dir.path().to_str().unwrap().to_string();

        let mut multi_builder = MultiTermBuilder::new(base_directory);

        // Build without adding any terms
        multi_builder.build().unwrap();

        // Should have no builders
        let builder_count = multi_builder.builders_iter_mut().count();
        assert_eq!(builder_count, 0);
    }
}
