use std::collections::HashMap;
use std::sync::RwLock;

use anyhow::Result;
use dashmap::mapref::one::Ref;
use dashmap::DashMap;

/// Per-document statistics for BM25 scoring.
/// Contains term frequency counts and document length.
#[derive(Debug, Clone, Default)]
pub struct DocStats {
    /// Document length (dl) in terms
    pub term_count: u32,
    /// Per-term TF counts for this doc (term -> count)
    pub term_frequencies: HashMap<String, u32>,
}

impl DocStats {
    /// Create a new DocStats with the given term count and frequencies.
    pub fn new(term_count: u32, term_frequencies: HashMap<String, u32>) -> Self {
        Self {
            term_count,
            term_frequencies,
        }
    }
}

/// Per-user BM25 statistics.
/// Tracks DF, TF, and document lengths for all documents of a single user.
#[derive(Debug, Default)]
pub struct UserBm25Stats {
    /// doc_id (point_id) -> per-document statistics
    pub doc_stats: HashMap<u32, DocStats>,
    /// term -> document frequency (DF) - number of docs containing this term
    pub doc_frequencies: HashMap<String, u32>,
    /// Total terms across all docs for this user (for avgdl calculation)
    pub total_term_count: u64,
}

impl UserBm25Stats {
    /// Create a new empty UserBm25Stats.
    pub fn new() -> Self {
        Self {
            doc_stats: HashMap::new(),
            doc_frequencies: HashMap::new(),
            total_term_count: 0,
        }
    }

    /// Add a document to the user's statistics.
    ///
    /// # Arguments
    /// * `doc_id` - The document ID (point_id)
    /// * `tokens` - The tokens/terms extracted from the document
    ///
    /// This updates:
    /// - doc_stats with TF counts for this document
    /// - doc_frequencies (DF) for each unique term
    /// - total_term_count for avgdl calculation
    pub fn add_document(&mut self, doc_id: u32, tokens: &[String]) {
        // Compute term frequencies for this document
        let term_frequencies = compute_term_frequencies(tokens);

        // Document length is the total number of tokens
        let term_count = tokens.len() as u32;

        // Create DocStats and store it
        let doc_stats = DocStats::new(term_count, term_frequencies.clone());
        self.doc_stats.insert(doc_id, doc_stats);

        // Update document frequencies (DF) for each unique term in this doc
        for term in term_frequencies.keys() {
            *self.doc_frequencies.entry(term.clone()).or_insert(0) += 1;
        }

        // Update total term count for avgdl calculation
        self.total_term_count += term_count as u64;
    }

    /// Get the number of documents for this user.
    pub fn num_documents(&self) -> usize {
        self.doc_stats.len()
    }

    /// Calculate the average document length (avgdl) for this user.
    /// Returns 0.0 if there are no documents.
    pub fn avg_document_length(&self) -> f64 {
        let num_docs = self.num_documents();
        if num_docs == 0 {
            0.0
        } else {
            self.total_term_count as f64 / num_docs as f64
        }
    }

    /// Get the document frequency (DF) for a specific term.
    pub fn get_df(&self, term: &str) -> u32 {
        self.doc_frequencies.get(term).copied().unwrap_or(0)
    }

    /// Get the term frequency (TF) for a specific term in a specific document.
    pub fn get_tf(&self, doc_id: u32, term: &str) -> u32 {
        self.doc_stats
            .get(&doc_id)
            .and_then(|stats| stats.term_frequencies.get(term).copied())
            .unwrap_or(0)
    }
}

/// Multi-user BM25 statistics collector.
/// Uses DashMap for concurrent access to per-user statistics.
#[derive(Debug, Default)]
pub struct MultiBm25StatsCollector {
    /// per-user stats using DashMap pattern (matching MultiTermBuilder)
    inner_stats: DashMap<u128, RwLock<UserBm25Stats>>,
}

impl MultiBm25StatsCollector {
    /// Create a new empty MultiBm25StatsCollector.
    pub fn new() -> Self {
        Self {
            inner_stats: DashMap::new(),
        }
    }

    /// Add a document to the statistics collector.
    ///
    /// # Arguments
    /// * `user_id` - The user ID
    /// * `point_id` - The document/point ID
    /// * `tokens` - The tokens extracted from the document
    ///
    /// # Returns
    /// * `Ok(())` on success
    pub fn add_document(&self, user_id: u128, point_id: u32, tokens: Vec<String>) -> Result<()> {
        // Get or create UserBm25Stats for this user using DashMap entry pattern
        let stats = self
            .inner_stats
            .entry(user_id)
            .or_insert_with(|| RwLock::new(UserBm25Stats::new()));

        // Acquire write lock and add the document
        let mut guard = stats.write().unwrap();
        guard.add_document(point_id, &tokens);

        Ok(())
    }

    /// Get the statistics for a specific user.
    ///
    /// # Arguments
    /// * `user_id` - The user ID
    ///
    /// # Returns
    /// * `Some(Ref<UserBm25Stats>)` if the user exists
    /// * `None` if the user has no statistics
    pub fn get_stats_for_user(
        &self,
        user_id: u128,
    ) -> Option<Ref<'_, u128, RwLock<UserBm25Stats>>> {
        self.inner_stats.get(&user_id)
    }

    /// Get all user IDs that have statistics.
    ///
    /// # Returns
    /// * `Vec<u128>` containing all user IDs
    pub fn get_all_user_ids(&self) -> Vec<u128> {
        self.inner_stats.iter().map(|entry| *entry.key()).collect()
    }

    /// Get the number of users with statistics.
    pub fn num_users(&self) -> usize {
        self.inner_stats.len()
    }

    /// Check if a user has statistics.
    pub fn has_user(&self, user_id: u128) -> bool {
        self.inner_stats.contains_key(&user_id)
    }
}

/// Compute term frequencies from a list of tokens.
fn compute_term_frequencies(tokens: &[String]) -> HashMap<String, u32> {
    let mut freqs = HashMap::new();
    for token in tokens {
        *freqs.entry(token.clone()).or_insert(0) += 1;
    }
    freqs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_user_single_doc() {
        let collector = MultiBm25StatsCollector::new();

        // Add a document with tokens ["apple", "banana"]
        collector
            .add_document(123u128, 0, vec!["apple".to_string(), "banana".to_string()])
            .unwrap();

        // Get user stats
        let user_stats_ref = collector.get_stats_for_user(123u128).unwrap();
        let user_stats = user_stats_ref.read().unwrap();

        // Verify: doc_stats[0].term_count == 2
        let doc_stat = user_stats.doc_stats.get(&0).unwrap();
        assert_eq!(doc_stat.term_count, 2);

        // Verify: term_frequencies["apple"] == 1
        assert_eq!(doc_stat.term_frequencies.get("apple"), Some(&1));
        assert_eq!(doc_stat.term_frequencies.get("banana"), Some(&1));

        // Verify total_term_count == 2
        assert_eq!(user_stats.total_term_count, 2);
    }

    #[test]
    fn test_df_tracking_multiple_docs() {
        let collector = MultiBm25StatsCollector::new();

        // Add doc0 with ["apple", "banana"]
        collector
            .add_document(123u128, 0, vec!["apple".to_string(), "banana".to_string()])
            .unwrap();

        // Add doc1 with ["apple", "cherry"]
        collector
            .add_document(123u128, 1, vec!["apple".to_string(), "cherry".to_string()])
            .unwrap();

        // Get user stats
        let user_stats_ref = collector.get_stats_for_user(123u128).unwrap();
        let user_stats = user_stats_ref.read().unwrap();

        // Verify: doc_frequencies["apple"] == 2 (appears in 2 docs)
        assert_eq!(user_stats.get_df("apple"), 2);

        // Verify: doc_frequencies["banana"] == 1
        assert_eq!(user_stats.get_df("banana"), 1);

        // Verify: doc_frequencies["cherry"] == 1
        assert_eq!(user_stats.get_df("cherry"), 1);

        // Verify: total_term_count == 4 (2 + 2)
        assert_eq!(user_stats.total_term_count, 4);

        // Verify: num_documents == 2
        assert_eq!(user_stats.num_documents(), 2);
    }

    #[test]
    fn test_multi_user_isolation() {
        let collector = MultiBm25StatsCollector::new();

        // User1: doc with ["apple"]
        collector
            .add_document(1u128, 0, vec!["apple".to_string()])
            .unwrap();

        // User2: doc with ["apple", "banana"]
        collector
            .add_document(2u128, 0, vec!["apple".to_string(), "banana".to_string()])
            .unwrap();

        // Get User1 stats
        let user1_stats_ref = collector.get_stats_for_user(1u128).unwrap();
        let user1_stats = user1_stats_ref.read().unwrap();

        // Verify User1: doc_frequencies["apple"] == 1
        assert_eq!(user1_stats.get_df("apple"), 1);
        // User1 should not have "banana"
        assert_eq!(user1_stats.get_df("banana"), 0);

        // Get User2 stats
        let user2_stats_ref = collector.get_stats_for_user(2u128).unwrap();
        let user2_stats = user2_stats_ref.read().unwrap();

        // Verify User2: doc_frequencies["apple"] == 1
        assert_eq!(user2_stats.get_df("apple"), 1);
        // Verify User2: doc_frequencies["banana"] == 1
        assert_eq!(user2_stats.get_df("banana"), 1);
    }

    #[test]
    fn test_term_frequency_counting() {
        let collector = MultiBm25StatsCollector::new();

        // Add doc with ["apple", "apple", "banana"]
        collector
            .add_document(
                123u128,
                0,
                vec![
                    "apple".to_string(),
                    "apple".to_string(),
                    "banana".to_string(),
                ],
            )
            .unwrap();

        // Get user stats
        let user_stats_ref = collector.get_stats_for_user(123u128).unwrap();
        let user_stats = user_stats_ref.read().unwrap();

        // Get the doc stats
        let doc_stat = user_stats.doc_stats.get(&0).unwrap();

        // Verify: term_frequencies["apple"] == 2
        assert_eq!(doc_stat.term_frequencies.get("apple"), Some(&2));

        // Verify: term_frequencies["banana"] == 1
        assert_eq!(doc_stat.term_frequencies.get("banana"), Some(&1));

        // Verify: term_count == 3 (total tokens)
        assert_eq!(doc_stat.term_count, 3);
    }

    #[test]
    fn test_total_term_count() {
        let collector = MultiBm25StatsCollector::new();

        // Add 2 docs with 3 and 5 tokens respectively
        collector
            .add_document(
                123u128,
                0,
                vec!["a".to_string(), "b".to_string(), "c".to_string()],
            )
            .unwrap();

        collector
            .add_document(
                123u128,
                1,
                vec![
                    "x".to_string(),
                    "y".to_string(),
                    "z".to_string(),
                    "w".to_string(),
                    "v".to_string(),
                ],
            )
            .unwrap();

        // Get user stats
        let user_stats_ref = collector.get_stats_for_user(123u128).unwrap();
        let user_stats = user_stats_ref.read().unwrap();

        // Verify: total_term_count == 8 (3 + 5)
        assert_eq!(user_stats.total_term_count, 8);

        // Verify avgdl == 4.0 (8 / 2 docs)
        assert_eq!(user_stats.avg_document_length(), 4.0);
    }

    #[test]
    fn test_empty_document() {
        let collector = MultiBm25StatsCollector::new();

        // Add empty document
        collector.add_document(123u128, 0, vec![]).unwrap();

        // Get user stats
        let user_stats_ref = collector.get_stats_for_user(123u128).unwrap();
        let user_stats = user_stats_ref.read().unwrap();

        // Get the doc stats
        let doc_stat = user_stats.doc_stats.get(&0).unwrap();

        // Verify: term_count == 0
        assert_eq!(doc_stat.term_count, 0);

        // Verify: term_frequencies is empty
        assert!(doc_stat.term_frequencies.is_empty());

        // Verify: total_term_count == 0
        assert_eq!(user_stats.total_term_count, 0);
    }

    #[test]
    fn test_get_all_user_ids() {
        let collector = MultiBm25StatsCollector::new();

        // Add documents for different users
        collector
            .add_document(1u128, 0, vec!["test".to_string()])
            .unwrap();
        collector
            .add_document(2u128, 0, vec!["test".to_string()])
            .unwrap();
        collector
            .add_document(3u128, 0, vec!["test".to_string()])
            .unwrap();

        // Get all user IDs
        let mut user_ids = collector.get_all_user_ids();
        user_ids.sort();

        // Verify we have exactly 3 users
        assert_eq!(user_ids.len(), 3);
        assert_eq!(user_ids, vec![1u128, 2u128, 3u128]);

        // Verify num_users()
        assert_eq!(collector.num_users(), 3);
    }

    #[test]
    fn test_has_user_and_nonexistent_user() {
        let collector = MultiBm25StatsCollector::new();

        // Initially, no users exist
        assert!(!collector.has_user(123u128));
        assert!(collector.get_stats_for_user(123u128).is_none());

        // Add a document for user 123
        collector
            .add_document(123u128, 0, vec!["test".to_string()])
            .unwrap();

        // Now user 123 should exist
        assert!(collector.has_user(123u128));
        assert!(collector.get_stats_for_user(123u128).is_some());

        // User 456 still doesn't exist
        assert!(!collector.has_user(456u128));
    }

    #[test]
    fn test_get_tf_helper() {
        let collector = MultiBm25StatsCollector::new();

        // Add document with repeated terms
        collector
            .add_document(
                123u128,
                0,
                vec![
                    "apple".to_string(),
                    "apple".to_string(),
                    "banana".to_string(),
                ],
            )
            .unwrap();

        // Get user stats
        let user_stats_ref = collector.get_stats_for_user(123u128).unwrap();
        let user_stats = user_stats_ref.read().unwrap();

        // Test get_tf helper
        assert_eq!(user_stats.get_tf(0, "apple"), 2);
        assert_eq!(user_stats.get_tf(0, "banana"), 1);
        assert_eq!(user_stats.get_tf(0, "cherry"), 0); // non-existent term
        assert_eq!(user_stats.get_tf(999, "apple"), 0); // non-existent doc
    }
}
