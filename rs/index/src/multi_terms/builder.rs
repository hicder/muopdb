use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::RwLock;

use anyhow::{anyhow, Result};
use dashmap::DashMap;

use crate::terms::builder::TermBuilder;

/// Builer struct for multiple users, each with their own TermBuilder.
pub struct MultiTermBuilder {
    /// Map from user ID to its corresponding TermBuilder.
    inner_builders: DashMap<u128, RwLock<TermBuilder>>,
    /// Whether the builders have been built.
    is_built: AtomicBool,
}

impl Default for MultiTermBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl MultiTermBuilder {
    pub fn new() -> Self {
        Self {
            inner_builders: DashMap::new(),
            is_built: AtomicBool::new(false),
        }
    }

    pub fn for_each_builder_mut<F>(&self, mut f: F)
    where
        F: FnMut(u128, &mut TermBuilder),
    {
        for mut entry in self.inner_builders.iter_mut() {
            let user_id = *entry.key();
            let mut builder = entry.value_mut().write().unwrap();
            f(user_id, &mut builder);
        }
    }

    pub fn get_user_ids(&self) -> Vec<u128> {
        self.inner_builders
            .iter()
            .map(|entry| *entry.key())
            .collect()
    }

    pub fn add(&self, user_id: u128, point_id: u32, key: String) -> Result<()> {
        // Try to get existing builder
        if let Some(builder_guard) = self.inner_builders.get(&user_id) {
            // Use existing builder
            let mut builder = builder_guard.write().unwrap();
            return builder.add(point_id, key);
        }

        // Create a new builder if it doesn't exist
        let new_builder = TermBuilder::new()?;

        // Insert new builder - need to handle potential race condition
        let builder_guard = self
            .inner_builders
            .entry(user_id)
            .or_insert_with(|| RwLock::new(new_builder));

        // Acquire write lock and add the term
        let mut builder = builder_guard.write().unwrap();
        builder.add(point_id, key)
    }

    pub fn build(&self) -> Result<()> {
        // Check if already built using atomic compare-and-swap
        if self
            .is_built
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return Err(anyhow!("MultiTermBuilder is already built"));
        }

        // Build all individual builders
        let mut build_errors = Vec::new();
        for mut entry in self.inner_builders.iter_mut() {
            let user_id = *entry.key();
            let mut builder = entry.value_mut().write().unwrap();
            if let Err(e) = builder.build() {
                build_errors.push((user_id, e));
            }
        }

        // If any build failed, reset the atomic flag and return error
        if !build_errors.is_empty() {
            self.is_built.store(false, Ordering::SeqCst);
            return Err(anyhow!(
                "Failed to build some term builders: {:?}",
                build_errors
            ));
        }

        Ok(())
    }

    pub fn is_built(&self) -> bool {
        self.is_built.load(Ordering::SeqCst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_term_builder() {
        let multi_builder = MultiTermBuilder::new();

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
        multi_builder.for_each_builder_mut(|user_id, builder| {
            builder_count += 1;

            match user_id {
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
        });

        // Should have exactly 2 users
        assert_eq!(builder_count, 2);
    }

    #[test]
    fn test_multi_term_builder_error_handling() {
        let multi_builder = MultiTermBuilder::new();

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
        let multi_builder = MultiTermBuilder::new();

        // Build without adding any terms
        multi_builder.build().unwrap();

        // Should have no builders
        let user_ids = multi_builder.get_user_ids();
        assert_eq!(user_ids.len(), 0);
    }
}
