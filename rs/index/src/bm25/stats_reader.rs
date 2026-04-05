use std::collections::HashMap;

use anyhow::{anyhow, Result};

use crate::bm25::stats_writer::SerializableUserStats;

/// View of per-user BM25 statistics for query-time scoring
pub struct UserBm25StatsView {
    pub num_docs: u32,
    pub total_terms: u64,
    pub avgdl: f64,
    pub doc_norms: Vec<u32>,           // Indexed by point_id
    pub term_df: HashMap<String, u32>, // term -> DF
}

impl UserBm25StatsView {
    /// Create a new empty stats view
    pub fn empty() -> Self {
        Self {
            num_docs: 0,
            total_terms: 0,
            avgdl: 0.0,
            doc_norms: Vec::new(),
            term_df: HashMap::new(),
        }
    }

    /// Get document length (dl) for a specific point_id
    pub fn get_doc_length(&self, point_id: u32) -> u32 {
        if (point_id as usize) < self.doc_norms.len() {
            self.doc_norms[point_id as usize]
        } else {
            0
        }
    }

    /// Get document frequency (DF) for a term
    pub fn get_term_df(&self, term: &str) -> u32 {
        self.term_df.get(term).copied().unwrap_or(0)
    }
}

/// Reader for loading BM25 statistics from disk
pub struct Bm25StatsReader {
    base_directory: String,
    user_stats: HashMap<u128, UserBm25StatsView>,
}

impl Bm25StatsReader {
    /// Load all statistics from the bm25_stats/ subdirectory
    pub fn new(base_directory: String) -> Result<Self> {
        let stats_dir = format!("{}/bm25_stats", base_directory);
        let mut user_stats = HashMap::new();

        // Check if stats directory exists
        if std::path::Path::new(&stats_dir).exists() {
            // Read all .rkyv files
            for entry in std::fs::read_dir(&stats_dir)? {
                let entry = entry?;
                let path = entry.path();

                if let Some(filename) = path.file_stem().and_then(|s| s.to_str()) {
                    if let Ok(user_id) = filename.parse::<u128>() {
                        // Read and deserialize using rkyv
                        let bytes = std::fs::read(&path)?;
                        let serializable =
                            rkyv::from_bytes::<SerializableUserStats, rkyv::rancor::Error>(&bytes)
                                .map_err(|e| anyhow!("rkyv deserialization failed: {}", e))?;

                        // Convert to view
                        let term_df: HashMap<String, u32> = serializable
                            .term_stats
                            .into_iter()
                            .map(|ts| (ts.term, ts.doc_freq))
                            .collect();

                        let view = UserBm25StatsView {
                            num_docs: serializable.num_docs,
                            total_terms: serializable.total_terms,
                            avgdl: serializable.avgdl,
                            doc_norms: serializable.doc_norms,
                            term_df,
                        };

                        user_stats.insert(user_id, view);
                    }
                }
            }
        }

        Ok(Self {
            base_directory,
            user_stats,
        })
    }

    /// Get statistics for a specific user
    pub fn get_stats_for_user(&self, user_id: u128) -> Option<&UserBm25StatsView> {
        self.user_stats.get(&user_id)
    }

    /// Get list of all users with statistics
    pub fn get_user_ids(&self) -> Vec<u128> {
        self.user_stats.keys().copied().collect()
    }

    /// Check if a user has statistics
    pub fn has_user(&self, user_id: u128) -> bool {
        self.user_stats.contains_key(&user_id)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use tempdir::TempDir;

    use super::*;
    use crate::bm25::stats_collector::MultiBm25StatsCollector;
    use crate::bm25::stats_writer::Bm25StatsWriter;

    #[test]
    fn test_roundtrip_serialization() {
        let tmp_dir = TempDir::new("test_bm25_roundtrip").unwrap();
        let base_dir = tmp_dir.path().to_str().unwrap().to_string();

        let collector = MultiBm25StatsCollector::new();

        // Add data for user 123
        collector
            .add_document(123u128, 0, vec!["apple".to_string(), "banana".to_string()])
            .unwrap();

        collector
            .add_document(123u128, 1, vec!["apple".to_string(), "cherry".to_string()])
            .unwrap();

        // Write stats
        let writer = Bm25StatsWriter::new(base_dir.clone());
        let mut max_point_ids = HashMap::new();
        max_point_ids.insert(123u128, 1);
        writer.write(&collector, &max_point_ids).unwrap();

        // Read back with reader
        let reader = Bm25StatsReader::new(base_dir).unwrap();
        let stats = reader.get_stats_for_user(123u128).unwrap();

        assert_eq!(stats.num_docs, 2);
        assert_eq!(stats.total_terms, 4);
        assert!((stats.avgdl - 2.0).abs() < f64::EPSILON);
        assert_eq!(stats.get_doc_length(0), 2);
        assert_eq!(stats.get_doc_length(1), 2);
        assert_eq!(stats.get_term_df("apple"), 2);
        assert_eq!(stats.get_term_df("banana"), 1);
        assert_eq!(stats.get_term_df("cherry"), 1);
    }

    #[test]
    fn test_get_doc_length() {
        let tmp_dir = TempDir::new("test_get_doc_length").unwrap();
        let base_dir = tmp_dir.path().to_str().unwrap().to_string();

        let collector = MultiBm25StatsCollector::new();

        // Add docs at various point_ids
        collector
            .add_document(456u128, 0, vec!["a".to_string(), "b".to_string()])
            .unwrap();
        collector
            .add_document(
                456u128,
                3,
                vec![
                    "c".to_string(),
                    "d".to_string(),
                    "e".to_string(),
                    "f".to_string(),
                ],
            )
            .unwrap();
        collector
            .add_document(456u128, 7, vec!["g".to_string()])
            .unwrap();

        let writer = Bm25StatsWriter::new(base_dir.clone());
        let mut max_point_ids = HashMap::new();
        max_point_ids.insert(456u128, 7);
        writer.write(&collector, &max_point_ids).unwrap();

        let reader = Bm25StatsReader::new(base_dir).unwrap();
        let stats = reader.get_stats_for_user(456u128).unwrap();

        assert_eq!(stats.get_doc_length(0), 2);
        assert_eq!(stats.get_doc_length(3), 4);
        assert_eq!(stats.get_doc_length(7), 1);
        assert_eq!(stats.get_doc_length(1), 0); // out of bounds returns 0
        assert_eq!(stats.get_doc_length(100), 0); // way out of bounds
    }

    #[test]
    fn test_get_term_df() {
        let tmp_dir = TempDir::new("test_get_term_df").unwrap();
        let base_dir = tmp_dir.path().to_str().unwrap().to_string();

        let collector = MultiBm25StatsCollector::new();

        // Add docs with known term frequencies
        collector
            .add_document(
                789u128,
                0,
                vec![
                    "apple".to_string(),
                    "apple".to_string(),
                    "banana".to_string(),
                ],
            )
            .unwrap();

        collector
            .add_document(789u128, 1, vec!["apple".to_string(), "cherry".to_string()])
            .unwrap();

        let writer = Bm25StatsWriter::new(base_dir.clone());
        let mut max_point_ids = HashMap::new();
        max_point_ids.insert(789u128, 1);
        writer.write(&collector, &max_point_ids).unwrap();

        let reader = Bm25StatsReader::new(base_dir).unwrap();
        let stats = reader.get_stats_for_user(789u128).unwrap();

        assert_eq!(stats.get_term_df("apple"), 2); // appears in 2 docs
        assert_eq!(stats.get_term_df("banana"), 1); // appears in 1 doc
        assert_eq!(stats.get_term_df("cherry"), 1); // appears in 1 doc
        assert_eq!(stats.get_term_df("unknown"), 0); // unknown term
    }

    #[test]
    fn test_missing_user() {
        let tmp_dir = TempDir::new("test_missing_user").unwrap();
        let base_dir = tmp_dir.path().to_str().unwrap().to_string();

        let collector = MultiBm25StatsCollector::new();
        collector
            .add_document(100u128, 0, vec!["test".to_string()])
            .unwrap();

        let writer = Bm25StatsWriter::new(base_dir.clone());
        let mut max_point_ids = HashMap::new();
        max_point_ids.insert(100u128, 0);
        writer.write(&collector, &max_point_ids).unwrap();

        let reader = Bm25StatsReader::new(base_dir).unwrap();

        // User 100 should exist
        assert!(reader.get_stats_for_user(100u128).is_some());
        assert!(reader.has_user(100u128));

        // User 99999 should not exist
        assert!(reader.get_stats_for_user(99999u128).is_none());
        assert!(!reader.has_user(99999u128));
    }

    #[test]
    fn test_missing_stats_directory() {
        let tmp_dir = TempDir::new("test_missing_stats_dir").unwrap();
        let base_dir = tmp_dir.path().to_str().unwrap().to_string();

        // Create reader without writing any stats (no bm25_stats directory)
        let reader = Bm25StatsReader::new(base_dir).unwrap();

        // Should succeed but have no users
        assert!(reader.get_user_ids().is_empty());
        assert!(!reader.has_user(123u128));
        assert!(reader.get_stats_for_user(123u128).is_none());
    }

    #[test]
    fn test_multi_user_stats() {
        let tmp_dir = TempDir::new("test_multi_user_stats").unwrap();
        let base_dir = tmp_dir.path().to_str().unwrap().to_string();

        let collector = MultiBm25StatsCollector::new();

        // User 1: doc with "apple banana"
        collector
            .add_document(1u128, 0, vec!["apple".to_string(), "banana".to_string()])
            .unwrap();

        // User 2: doc with "apple cherry"
        collector
            .add_document(2u128, 0, vec!["apple".to_string(), "cherry".to_string()])
            .unwrap();

        let writer = Bm25StatsWriter::new(base_dir.clone());
        let mut max_point_ids = HashMap::new();
        max_point_ids.insert(1u128, 0);
        max_point_ids.insert(2u128, 0);
        writer.write(&collector, &max_point_ids).unwrap();

        let reader = Bm25StatsReader::new(base_dir).unwrap();

        // Verify both users exist
        assert_eq!(reader.get_user_ids().len(), 2);
        assert!(reader.has_user(1u128));
        assert!(reader.has_user(2u128));

        // Verify per-user isolation
        let user1_stats = reader.get_stats_for_user(1u128).unwrap();
        assert_eq!(user1_stats.get_term_df("apple"), 1);
        assert_eq!(user1_stats.get_term_df("banana"), 1);
        assert_eq!(user1_stats.get_term_df("cherry"), 0); // User 1 doesn't have cherry

        let user2_stats = reader.get_stats_for_user(2u128).unwrap();
        assert_eq!(user2_stats.get_term_df("apple"), 1);
        assert_eq!(user2_stats.get_term_df("banana"), 0); // User 2 doesn't have banana
        assert_eq!(user2_stats.get_term_df("cherry"), 1);
    }
}
