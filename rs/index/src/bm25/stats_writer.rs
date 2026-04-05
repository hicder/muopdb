use std::collections::HashMap;

use anyhow::Result;
use rkyv::{Archive, Deserialize, Serialize};

use crate::bm25::stats_collector::{MultiBm25StatsCollector, UserBm25Stats};

/// Per-term statistics for serialization (DF only - TF is per-doc)
#[derive(Archive, Serialize, Deserialize, Debug, Clone)]
pub struct TermStat {
    pub term: String,
    pub doc_freq: u32, // DF: number of docs containing this term
}

/// Per-user BM25 statistics for rkyv serialization
#[derive(Archive, Serialize, Deserialize, Debug, Clone)]
pub struct SerializableUserStats {
    pub num_docs: u32,
    pub total_terms: u64,
    pub avgdl: f64,
    pub doc_norms: Vec<u32>, // Indexed by point_id (doc_norms[point_id] = dl)
    pub term_stats: Vec<TermStat>, // DF for each term
}

impl SerializableUserStats {
    /// Convert from runtime UserBm25Stats to serializable form
    pub fn from_user_stats(user_stats: &UserBm25Stats, max_point_id: u32) -> Self {
        let num_docs = user_stats.doc_stats.len() as u32;
        let total_terms = user_stats.total_term_count;
        let avgdl = if num_docs > 0 {
            total_terms as f64 / num_docs as f64
        } else {
            0.0
        };

        // Build doc_norms array - indexed by point_id
        // Size to max_point_id + 1 for direct indexing
        let mut doc_norms = vec![0u32; (max_point_id + 1) as usize];
        for (point_id, doc_stats) in &user_stats.doc_stats {
            doc_norms[*point_id as usize] = doc_stats.term_count;
        }

        // Convert doc_frequencies HashMap to TermStat vec
        let term_stats: Vec<TermStat> = user_stats
            .doc_frequencies
            .iter()
            .map(|(term, df)| TermStat {
                term: term.clone(),
                doc_freq: *df,
            })
            .collect();

        Self {
            num_docs,
            total_terms,
            avgdl,
            doc_norms,
            term_stats,
        }
    }
}

/// Writer for persisting BM25 statistics per segment
pub struct Bm25StatsWriter {
    base_directory: String,
}

impl Bm25StatsWriter {
    pub fn new(base_directory: String) -> Self {
        Self { base_directory }
    }

    /// Write statistics for all users to the bm25_stats/ subdirectory
    pub fn write(
        &self,
        collector: &MultiBm25StatsCollector,
        max_point_ids: &HashMap<u128, u32>,
    ) -> Result<()> {
        // Create bm25_stats subdirectory
        let stats_dir = format!("{}/bm25_stats", self.base_directory);
        std::fs::create_dir_all(&stats_dir)?;

        // Write stats for each user
        for user_id in collector.get_all_user_ids() {
            if let Some(user_stats_ref) = collector.get_stats_for_user(user_id) {
                let user_stats = user_stats_ref.read().unwrap();
                let max_point_id = max_point_ids.get(&user_id).copied().unwrap_or(0);
                let serializable =
                    SerializableUserStats::from_user_stats(&user_stats, max_point_id);

                // Serialize with rkyv
                let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&serializable)
                    .map_err(|e| anyhow::anyhow!("rkyv serialization failed: {}", e))?;

                // Write to file: bm25_stats/{user_id}.rkyv
                let file_path = format!("{}/{}.rkyv", stats_dir, user_id);
                std::fs::write(&file_path, &bytes)?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use tempdir::TempDir;

    use super::*;
    use crate::bm25::stats_collector::MultiBm25StatsCollector;

    #[test]
    fn test_avgdl_computation() {
        let collector = MultiBm25StatsCollector::new();

        // Add 2 docs: doc0 (5 terms), doc1 (3 terms)
        collector
            .add_document(
                123u128,
                0,
                vec![
                    "a".to_string(),
                    "b".to_string(),
                    "c".to_string(),
                    "d".to_string(),
                    "e".to_string(),
                ],
            )
            .unwrap();

        collector
            .add_document(
                123u128,
                1,
                vec!["x".to_string(), "y".to_string(), "z".to_string()],
            )
            .unwrap();

        let user_stats_ref = collector.get_stats_for_user(123u128).unwrap();
        let user_stats = user_stats_ref.read().unwrap();
        let serializable = SerializableUserStats::from_user_stats(&user_stats, 1);

        assert_eq!(serializable.num_docs, 2);
        assert_eq!(serializable.total_terms, 8);
        assert_eq!(serializable.avgdl, 4.0);
    }

    #[test]
    fn test_doc_norms_array() {
        let collector = MultiBm25StatsCollector::new();

        // Add docs at point_id 0 (dl=5) and 5 (dl=10)
        collector
            .add_document(
                123u128,
                0,
                vec![
                    "a".to_string(),
                    "b".to_string(),
                    "c".to_string(),
                    "d".to_string(),
                    "e".to_string(),
                ],
            )
            .unwrap();

        collector
            .add_document(
                123u128,
                5,
                vec![
                    "1".to_string(),
                    "2".to_string(),
                    "3".to_string(),
                    "4".to_string(),
                    "5".to_string(),
                    "6".to_string(),
                    "7".to_string(),
                    "8".to_string(),
                    "9".to_string(),
                    "10".to_string(),
                ],
            )
            .unwrap();

        let user_stats_ref = collector.get_stats_for_user(123u128).unwrap();
        let user_stats = user_stats_ref.read().unwrap();
        let serializable = SerializableUserStats::from_user_stats(&user_stats, 5);

        // doc_norms should have length 6 (indices 0-5)
        assert_eq!(serializable.doc_norms.len(), 6);
        assert_eq!(serializable.doc_norms[0], 5);
        assert_eq!(serializable.doc_norms[5], 10);
        assert_eq!(serializable.doc_norms[1], 0);
        assert_eq!(serializable.doc_norms[4], 0);
    }

    #[test]
    fn test_term_stats_df() {
        let collector = MultiBm25StatsCollector::new();

        // Add docs with overlapping terms
        collector
            .add_document(123u128, 0, vec!["apple".to_string(), "banana".to_string()])
            .unwrap();

        collector
            .add_document(123u128, 1, vec!["apple".to_string(), "cherry".to_string()])
            .unwrap();

        collector
            .add_document(
                123u128,
                2,
                vec![
                    "apple".to_string(),
                    "banana".to_string(),
                    "date".to_string(),
                ],
            )
            .unwrap();

        let user_stats_ref = collector.get_stats_for_user(123u128).unwrap();
        let user_stats = user_stats_ref.read().unwrap();
        let serializable = SerializableUserStats::from_user_stats(&user_stats, 2);

        // Convert term_stats to HashMap for easier testing
        let df_map: HashMap<String, u32> = serializable
            .term_stats
            .iter()
            .map(|ts| (ts.term.clone(), ts.doc_freq))
            .collect();

        assert_eq!(df_map.get("apple"), Some(&3)); // appears in 3 docs
        assert_eq!(df_map.get("banana"), Some(&2)); // appears in 2 docs
        assert_eq!(df_map.get("cherry"), Some(&1)); // appears in 1 doc
        assert_eq!(df_map.get("date"), Some(&1)); // appears in 1 doc
    }

    #[test]
    fn test_stats_writer_creates_directory() {
        let tmp_dir = TempDir::new("test_bm25_stats_writer").unwrap();
        let base_dir = tmp_dir.path().to_str().unwrap().to_string();

        let collector = MultiBm25StatsCollector::new();

        // Add document for user 123
        collector
            .add_document(123u128, 0, vec!["apple".to_string(), "banana".to_string()])
            .unwrap();

        // Create writer and write
        let writer = Bm25StatsWriter::new(base_dir.clone());
        let mut max_point_ids = HashMap::new();
        max_point_ids.insert(123u128, 0);
        writer.write(&collector, &max_point_ids).unwrap();

        // Verify bm25_stats directory exists
        let stats_dir = format!("{}/bm25_stats", base_dir);
        assert!(std::path::Path::new(&stats_dir).exists());

        // Verify user file exists
        let user_file = format!("{}/123.rkyv", stats_dir);
        assert!(std::path::Path::new(&user_file).exists());

        // Verify file has content (not empty)
        let file_bytes = std::fs::read(&user_file).unwrap();
        assert!(!file_bytes.is_empty(), "Stats file should not be empty");
    }
}
