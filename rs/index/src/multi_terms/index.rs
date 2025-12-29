use std::sync::Arc;

use anyhow::{anyhow, Result};
use dashmap::DashMap;
use utils::file_io::env::Env;

use crate::terms::index::TermIndex;
use crate::terms::term_index_info::TermIndexInfoHashTable;

pub struct MultiTermIndex {
    /// Map of user ID to their [`TermIndex`]
    term_indexes: DashMap<u128, Arc<TermIndex>>,
    /// Hash table holding offsets and lengths for all users
    user_index_info: TermIndexInfoHashTable,
    /// Base directory for the term indexes
    base_directory: String,
    /// Optional Env for async file I/O
    env: Option<Arc<Box<dyn Env>>>,
}

impl MultiTermIndex {
    /// Load the MultiTermIndex and the TermIndexInfo hash table
    pub fn new(base_directory: String) -> Result<Self> {
        // Load the user index info hash table
        let info_file_path = format!("{}/user_term_index_info", base_directory);
        let user_index_info = TermIndexInfoHashTable::load(info_file_path)?;

        Ok(Self {
            base_directory,
            term_indexes: DashMap::new(),
            user_index_info,
            env: None,
        })
    }

    /// Load the MultiTermIndex with an Env for async file I/O
    pub async fn new_with_env(
        base_directory: String,
        env: Arc<Box<dyn Env>>,
    ) -> Result<Self> {
        // Load the user index info hash table
        let info_file_path = format!("{}/user_term_index_info", base_directory);
        let user_index_info = TermIndexInfoHashTable::load(info_file_path)?;

        Ok(Self {
            base_directory,
            term_indexes: DashMap::new(),
            user_index_info,
            env: Some(env),
        })
    }

    /// Lazily load or return cached TermIndex for a user
    /// Errors: if user not found or TermIndex creation fails
    pub fn get_or_create_index(&self, user_id: u128) -> Result<Arc<TermIndex>> {
        // Try to get from cache first
        if let Some(term_index) = self.term_indexes.get(&user_id) {
            return Ok(term_index.clone());
        }

        // Not in cache, create new one
        let index_info = self
            .user_index_info
            .hash_table
            .get(&user_id)
            .ok_or_else(|| anyhow!("User not found"))?;

        let combined_path = format!("{}/combined", self.base_directory);
        let term_index = TermIndex::new(
            combined_path,
            index_info.offset as usize,
            index_info.length as usize,
        )
        .map_err(|e| anyhow!("Failed to create TermIndex: {e}"))?;

        let term_index_arc = Arc::new(term_index);
        self.term_indexes.insert(user_id, term_index_arc.clone());
        Ok(term_index_arc)
    }

    /// Lazily load or return cached TermIndex for a user asynchronously, with file_io support if Env is available
    pub async fn get_or_create_index_async(&self, user_id: u128) -> Result<Arc<TermIndex>> {
        // Try to get from cache first
        if let Some(term_index) = self.term_indexes.get(&user_id) {
            return Ok(term_index.clone());
        }

        // Not in cache, create new one
        let index_info = self
            .user_index_info
            .hash_table
            .get(&user_id)
            .ok_or_else(|| anyhow!("User not found"))?;

        let combined_path = format!("{}/combined", self.base_directory);

        let term_index = if let Some(env) = &self.env {
            let file_io = env.open(&combined_path).await?.file_io;
            TermIndex::new_with_file_io(
                file_io,
                combined_path,
                index_info.offset as usize,
                index_info.length as usize,
            )
            .await
            .map_err(|e| anyhow!("Failed to create TermIndex with file_io: {e}"))?
        } else {
            TermIndex::new(
                combined_path,
                index_info.offset as usize,
                index_info.length as usize,
            )
            .map_err(|e| anyhow!("Failed to create TermIndex: {e}"))?
        };

        let term_index_arc = Arc::new(term_index);
        self.term_indexes.insert(user_id, term_index_arc.clone());
        Ok(term_index_arc)
    }

    /// Retrieve the term ID for a given user and term string
    /// Will create/load the TermIndex for the user if not already cached
    /// Errors: if user or term not found or TermIndex creation fails
    pub fn get_term_id_for_user(&self, user_id: u128, term: &str) -> Result<u64> {
        let term_index = self.get_or_create_index(user_id)?;
        term_index
            .get_term_id(term)
            .ok_or_else(|| anyhow!("Term not found"))
    }

    /// Remove and return the TermIndex for a given user
    /// Will create/load the TermIndex for the user if not already cached
    /// Errors: if user not found or TermIndex creation fails
    pub fn take_index_for_user(&self, user_id: u128) -> Result<Arc<TermIndex>> {
        self.get_or_create_index(user_id)?;
        self.term_indexes
            .remove(&user_id)
            .map(|(_, term_index)| term_index)
            .ok_or_else(|| anyhow!("User not found"))
    }

    #[cfg(test)]
    pub fn term_index_info(&self) -> &TermIndexInfoHashTable {
        &self.user_index_info
    }

    /// Iterates all (term, point_id) pairs for a given user.
    pub fn iter_term_point_pairs_for_user(
        &self,
        user_id: u128,
    ) -> Result<Box<dyn Iterator<Item = (String, u32)> + '_>> {
        let term_index = self.get_or_create_index(user_id)?;
        let pairs: Vec<(String, u32)> = term_index.iter_term_point_pairs().collect();
        Ok(Box::new(pairs.into_iter()))
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;
    use std::sync::Arc;

    use compression::AsyncIntSeqIterator;
    use tempdir::TempDir;
    use utils::file_io::env::{DefaultEnv, Env, EnvConfig, FileType};

    use super::MultiTermIndex;
    use crate::multi_terms::builder::MultiTermBuilder;
    use crate::multi_terms::writer::MultiTermWriter;

    fn build_and_write_index(base_dir: &str) -> (Vec<u128>, MultiTermIndex) {
        let mut multi_builder = MultiTermBuilder::new();
        let user1 = 1001u128;
        let user2 = 2002u128;
        let user3 = 3003u128;

        // User 1 terms
        multi_builder
            .add(user1, 10, "apple:red".to_string())
            .unwrap();
        multi_builder
            .add(user1, 20, "banana:yellow".to_string())
            .unwrap();
        multi_builder
            .add(user1, 30, "apple:green".to_string())
            .unwrap();

        // User 2 terms
        multi_builder
            .add(user2, 11, "car:toyota".to_string())
            .unwrap();
        multi_builder
            .add(user2, 22, "car:honda".to_string())
            .unwrap();
        multi_builder
            .add(user2, 33, "bike:yamaha".to_string())
            .unwrap();

        // User 3 single term
        multi_builder
            .add(user3, 1, "test:value".to_string())
            .unwrap();

        multi_builder.build().unwrap();
        let writer = MultiTermWriter::new(base_dir.to_string());
        writer.write(&mut multi_builder).unwrap();

        let index = MultiTermIndex::new(base_dir.to_string()).unwrap();
        (vec![user1, user2, user3], index)
    }

    #[test]
    fn test_multi_term_index_roundtrip() {
        let tmp = TempDir::new("multi_term_index_roundtrip").unwrap();
        let base_dir = tmp.path().to_str().unwrap().to_string();

        let (users, index) = build_and_write_index(&base_dir);
        let user1 = users[0];
        let user2 = users[1];

        // Term IDs should differ within the same user
        let id_apple_red = index.get_term_id_for_user(user1, "apple:red").unwrap();
        let id_banana = index.get_term_id_for_user(user1, "banana:yellow").unwrap();
        assert_ne!(id_apple_red, id_banana);

        // Verify isolation of lookup between users
        assert!(
            index
                .get_term_id_for_user(user1, "car:toyota")
                .unwrap_err()
                .to_string()
                == "Term not found",
            "User1 should not see User2's terms"
        );
        assert!(
            index
                .get_term_id_for_user(user2, "apple:red")
                .unwrap_err()
                .to_string()
                == "Term not found",
            "User2 should not see User1's terms"
        );

        // Check that we can open posting list iterator and it matches builder data
        let term_index_apple = index.get_or_create_index(user1).unwrap();
        let pl_apple: Vec<_> = term_index_apple
            .get_posting_list_iterator(id_apple_red)
            .unwrap()
            .collect();
        assert_eq!(pl_apple, vec![10]);

        let term_id = index.get_term_id_for_user(user2, "bike:yamaha").unwrap();
        let term_index_bike = index.get_or_create_index(user2).unwrap();
        let pl_bike: Vec<_> = term_index_bike
            .get_posting_list_iterator(term_id)
            .unwrap()
            .collect();
        assert_eq!(pl_bike, vec![33]);
    }

    #[test]
    fn test_multi_term_index_empty() {
        let tmp = TempDir::new("multi_term_index_empty").unwrap();
        let base_dir = tmp.path().to_str().unwrap().to_string();

        let mut builder = MultiTermBuilder::new();
        builder.build().unwrap();

        let writer = MultiTermWriter::new(base_dir.clone());
        writer.write(&mut builder).unwrap();

        let combined_path = format!("{}/combined", base_dir);
        assert!(Path::new(&combined_path).exists());
        let len = fs::metadata(&combined_path).unwrap().len();
        assert_eq!(len, 0, "Combined file should be empty");

        let index_info_path = format!("{}/user_term_index_info", base_dir);
        assert!(Path::new(&index_info_path).exists());

        let index = MultiTermIndex::new(base_dir.clone()).unwrap();
        // The hash table should be empty
        assert_eq!(index.term_index_info().hash_table.len(), 0);
        // Attempting to get a user should fail
        assert!(index.get_or_create_index(123u128).is_err());
    }

    #[test]
    fn test_multi_term_index_alignment_and_ranges() {
        let tmp = TempDir::new("multi_term_index_alignment").unwrap();
        let base_dir = tmp.path().to_str().unwrap().to_string();

        let (users, index) = build_and_write_index(&base_dir);
        let info = index.term_index_info();

        // Combined file size
        let combined_len = fs::metadata(format!("{}/combined", base_dir))
            .unwrap()
            .len() as u64;

        // Verify alignment and non-overlap
        let mut last_end = 0;
        let mut offsets = vec![];

        for user_id in &users {
            let entry = info.hash_table.get(user_id).unwrap();
            assert_eq!(entry.offset % 8, 0, "Offset not 8-byte aligned");
            assert!(entry.offset + entry.length <= combined_len);
            assert!(entry.offset >= last_end, "Overlapping regions");
            last_end = entry.offset + entry.length;
            offsets.push(entry.offset);
        }

        // Ensure deterministic order (sorted offsets)
        let mut sorted_offsets = offsets.clone();
        sorted_offsets.sort();
        assert_eq!(offsets, sorted_offsets);
    }

    #[test]
    fn test_multi_term_index_lazy_load_cache() {
        let tmp = TempDir::new("multi_term_index_lazy_cache").unwrap();
        let base_dir = tmp.path().to_str().unwrap().to_string();

        let (users, index) = build_and_write_index(&base_dir);
        let user1 = users[0];

        // First call should load
        let first = Arc::as_ptr(&index.get_or_create_index(user1).unwrap());
        // Second call should reuse
        let second = Arc::as_ptr(&index.get_or_create_index(user1).unwrap());
        assert_eq!(first, second, "Should return cached TermIndex reference");

        // Should fail for unknown user
        let unknown_user = 99999u128;
        assert!(index.get_or_create_index(unknown_user).is_err());
    }

    #[tokio::test]
    async fn test_multi_term_index_async_with_env() {
        let tmp = TempDir::new("multi_term_index_async").unwrap();
        let base_dir = tmp.path().to_str().unwrap().to_string();

        let (users, _) = build_and_write_index(&base_dir);
        let user1 = users[0];

        let config = EnvConfig {
            file_type: FileType::CachedStandard,
            ..EnvConfig::default()
        };
        let env: Arc<Box<dyn Env>> = Arc::new(Box::new(DefaultEnv::new(config)));

        let index = MultiTermIndex::new_with_env(base_dir, env).await.unwrap();
        let term_index = index.get_or_create_index_async(user1).await.unwrap();

        // Verify it works by trying block-based iterator
        let term_id = index.get_term_id_for_user(user1, "apple:red").unwrap();
        let mut it = term_index
            .get_posting_list_iterator_block_based(term_id)
            .await
            .unwrap();
        assert_eq!(it.next().await.unwrap(), Some(10u32));
        assert_eq!(it.next().await.unwrap(), None);
    }
}
