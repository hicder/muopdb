use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{anyhow, Result};

use crate::bm25::config::BM25Config;
use crate::bm25::query::{Bm25Query, Bm25SearchResult, Bm25SearchResults};
use crate::bm25::scorer::BM25Scorer;
use crate::bm25::stats_reader::{Bm25StatsReader, UserBm25StatsView};
use crate::multi_terms::index::MultiTermIndex;
use crate::tokenizer::TokenizerPipeline;

/// BM25 Searcher for executing text queries and computing relevance scores
///
/// This is the core search component that:
/// - Tokenizes query text using the same pipeline as indexing (TokenizerPipeline)
/// - Retrieves document statistics for IDF calculation
/// - Iterates posting lists to get term frequencies
/// - Computes BM25 scores using the BM25Scorer
pub struct Bm25Searcher {
    stats_reader: Bm25StatsReader,
    multi_term_index: Arc<MultiTermIndex>,
    scorer: BM25Scorer,
    tokenizer: TokenizerPipeline,
}

impl Bm25Searcher {
    /// Create a new BM25 searcher
    ///
    /// # Arguments
    /// * `stats_dir` - Directory containing BM25 statistics (bm25_stats subdirectory will be used)
    /// * `terms_dir` - Directory containing the MultiTermIndex
    /// * `config` - BM25 scoring configuration (k1, b parameters)
    ///
    /// # Returns
    /// A configured Bm25Searcher ready to execute queries
    pub fn new(stats_dir: String, terms_dir: String, config: BM25Config) -> Result<Self> {
        let stats_reader = Bm25StatsReader::new(stats_dir)?;
        let multi_term_index = Arc::new(MultiTermIndex::new(terms_dir)?);
        let scorer = BM25Scorer::new(config);
        let tokenizer = TokenizerPipeline::english();

        Ok(Self {
            stats_reader,
            multi_term_index,
            scorer,
            tokenizer,
        })
    }

    /// Execute a search query and return ranked results
    ///
    /// # Arguments
    /// * `query` - The BM25 search query containing text, user_id, and max_results
    ///
    /// # Returns
    /// Search results sorted by BM25 score (highest first)
    pub fn search(&self, query: &Bm25Query) -> Result<Bm25SearchResults> {
        // Get user statistics
        let stats = self
            .stats_reader
            .get_stats_for_user(query.user_id)
            .ok_or_else(|| anyhow!("No statistics found for user {}", query.user_id))?;

        // Tokenize query text using the same pipeline as indexing
        let terms = self.tokenizer.tokenize(&query.text);

        // Handle empty query
        if terms.is_empty() {
            return Ok(Bm25SearchResults::empty());
        }

        // Score documents based on query terms
        let scores = if terms.len() == 1 {
            self.score_single_term(query.user_id, &terms[0], stats)?
        } else {
            self.score_multi_term(query.user_id, &terms, stats)?
        };

        // Convert scores to results
        let mut results: Vec<Bm25SearchResult> = scores
            .into_iter()
            .map(|(point_id, score)| Bm25SearchResult::new(point_id, score))
            .collect();

        // Sort by score descending (highest first)
        results.sort();

        // Limit to max_results
        let total_hits = results.len();
        if results.len() > query.max_results {
            results.truncate(query.max_results);
        }

        Ok(Bm25SearchResults::with_total_matches(results, total_hits))
    }

    /// Score documents for a single query term
    fn score_single_term(
        &self,
        user_id: u128,
        term: &str,
        stats: &UserBm25StatsView,
    ) -> Result<HashMap<u32, f32>> {
        let mut scores = HashMap::new();

        // Get term ID from MultiTermIndex
        let term_id = match self.multi_term_index.get_term_id_for_user(user_id, term) {
            Ok(id) => id,
            Err(_) => return Ok(scores),
        };

        // Get posting list iterator for the term
        let term_index = self.multi_term_index.get_or_create_index(user_id)?;
        let posting_list = match term_index.get_posting_list_iterator(term_id) {
            Ok(iter) => iter,
            Err(_) => return Ok(scores),
        };

        // Get document frequency for IDF calculation
        let doc_freq = stats.get_term_df(term);
        let total_docs = stats.num_docs;
        let avgdl = stats.avgdl;

        // Score each document in the posting list
        for point_id in posting_list {
            let doc_length = stats.get_doc_length(point_id);
            let term_freq = 1u32;

            let score = self
                .scorer
                .score(term_freq, doc_length, avgdl, doc_freq, total_docs);

            scores.insert(point_id, score);
        }

        Ok(scores)
    }

    /// Score documents for multiple query terms with score aggregation
    fn score_multi_term(
        &self,
        user_id: u128,
        terms: &[String],
        stats: &UserBm25StatsView,
    ) -> Result<HashMap<u32, f32>> {
        let mut aggregated_scores: HashMap<u32, f32> = HashMap::new();

        for term in terms {
            let term_scores = self.score_single_term(user_id, term, stats)?;
            for (point_id, score) in term_scores {
                *aggregated_scores.entry(point_id).or_insert(0.0) += score;
            }
        }

        Ok(aggregated_scores)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use tempdir::TempDir;

    use super::*;
    use crate::bm25::stats_collector::MultiBm25StatsCollector;
    use crate::bm25::stats_writer::Bm25StatsWriter;
    use crate::multi_terms::builder::MultiTermBuilder;
    use crate::multi_terms::writer::MultiTermWriter;

    fn setup_test_data(_base_dir: &str) -> (u128, MultiBm25StatsCollector, HashMap<u128, u32>) {
        let user_id = 123u128;
        let collector = MultiBm25StatsCollector::new();

        // Add documents with known terms.
        // Use 5 docs so terms appearing in 1-2 docs have DF <= 40% (<50% threshold for positive IDF)

        // Doc 0: "apple banana" (2 terms)
        collector
            .add_document(user_id, 0, vec!["apple".to_string(), "banana".to_string()])
            .unwrap();

        // Doc 1: "apple cherry" (2 terms)
        collector
            .add_document(user_id, 1, vec!["apple".to_string(), "cherry".to_string()])
            .unwrap();

        // Doc 2: "banana cherry" (2 terms) - apple not here
        collector
            .add_document(user_id, 2, vec!["banana".to_string(), "cherry".to_string()])
            .unwrap();

        // Doc 3: "apple banana date" (3 terms) - date only in 2 docs (40%)
        collector
            .add_document(
                user_id,
                3,
                vec![
                    "apple".to_string(),
                    "banana".to_string(),
                    "date".to_string(),
                ],
            )
            .unwrap();

        // Doc 4: "cherry date" (2 terms) - date in docs 3, 4 = 2/5 = 40%
        collector
            .add_document(user_id, 4, vec!["cherry".to_string(), "date".to_string()])
            .unwrap();

        let mut max_point_ids = HashMap::new();
        max_point_ids.insert(user_id, 4);

        (user_id, collector, max_point_ids)
    }

    fn build_test_index(base_dir: &str, user_id: u128) {
        let multi_builder = MultiTermBuilder::new();

        multi_builder.add(user_id, 0, "apple".to_string()).unwrap();
        multi_builder.add(user_id, 0, "banana".to_string()).unwrap();
        multi_builder.add(user_id, 1, "apple".to_string()).unwrap();
        multi_builder.add(user_id, 1, "cherry".to_string()).unwrap();
        multi_builder.add(user_id, 2, "banana".to_string()).unwrap();
        multi_builder.add(user_id, 2, "cherry".to_string()).unwrap();
        multi_builder.add(user_id, 3, "apple".to_string()).unwrap();
        multi_builder.add(user_id, 3, "banana".to_string()).unwrap();
        multi_builder.add(user_id, 3, "date".to_string()).unwrap();
        multi_builder.add(user_id, 4, "cherry".to_string()).unwrap();
        multi_builder.add(user_id, 4, "date".to_string()).unwrap();

        multi_builder.build().unwrap();

        let terms_dir = format!("{}/terms", base_dir);
        let writer = MultiTermWriter::new(terms_dir);
        writer.write(&multi_builder).unwrap();
    }

    #[test]
    fn test_single_term_search() {
        let tmp_dir = TempDir::new("test_bm25_single").unwrap();
        let base_dir = tmp_dir.path().to_str().unwrap().to_string();
        let (user_id, collector, max_point_ids) = setup_test_data(&base_dir);

        let stats_writer = Bm25StatsWriter::new(base_dir.clone());
        stats_writer.write(&collector, &max_point_ids).unwrap();
        build_test_index(&base_dir, user_id);

        let config = BM25Config::default();
        let searcher =
            Bm25Searcher::new(base_dir.clone(), format!("{}/terms", base_dir), config).unwrap();

        // Search for "date" - appears in 2 of 5 docs (40%), has positive IDF
        let query = Bm25Query::new("date".to_string(), user_id);
        let results = searcher.search(&query).unwrap();

        assert_eq!(results.total_hits, 2, "Should find 2 documents with 'date'");
        assert!(!results.is_empty());
        assert!(results.results[0].score > 0.0, "Score should be positive");
    }

    #[test]
    fn test_multi_term_score_aggregation() {
        let tmp_dir = TempDir::new("test_bm25_multi").unwrap();
        let base_dir = tmp_dir.path().to_str().unwrap().to_string();
        let (user_id, collector, max_point_ids) = setup_test_data(&base_dir);

        let stats_writer = Bm25StatsWriter::new(base_dir.clone());
        stats_writer.write(&collector, &max_point_ids).unwrap();
        build_test_index(&base_dir, user_id);

        let config = BM25Config::default();
        let searcher =
            Bm25Searcher::new(base_dir.clone(), format!("{}/terms", base_dir), config).unwrap();

        // Single term search for "date"
        let single_query = Bm25Query::new("date".to_string(), user_id);
        let single_results = searcher.search(&single_query).unwrap();
        let single_doc3_score = single_results
            .results
            .iter()
            .find(|r| r.point_id == 3)
            .map(|r| r.score)
            .unwrap_or(0.0);

        // Multi-term search for "date cherry" - doc 3 has "date", doc 3 and 4 have "cherry"
        let multi_query = Bm25Query::new("date cherry".to_string(), user_id);
        let multi_results = searcher.search(&multi_query).unwrap();

        // Doc 3 has both terms, so should have higher score than single-term
        let multi_doc3_score = multi_results
            .results
            .iter()
            .find(|r| r.point_id == 3)
            .map(|r| r.score)
            .unwrap_or(0.0);

        assert!(
            multi_doc3_score >= single_doc3_score,
            "Multi-term score ({}) should be >= single-term score ({})",
            multi_doc3_score,
            single_doc3_score
        );
    }

    #[test]
    fn test_query_tokenization() {
        let tmp_dir = TempDir::new("test_bm25_tok").unwrap();
        let base_dir = tmp_dir.path().to_str().unwrap().to_string();
        let (user_id, collector, max_point_ids) = setup_test_data(&base_dir);

        let stats_writer = Bm25StatsWriter::new(base_dir.clone());
        stats_writer.write(&collector, &max_point_ids).unwrap();
        build_test_index(&base_dir, user_id);

        let config = BM25Config::default();
        let searcher =
            Bm25Searcher::new(base_dir.clone(), format!("{}/terms", base_dir), config).unwrap();

        // Query "The running dogs" -> tokenized to "run", "dog" after stemming and stop word removal
        // But "run" and "dog" are not in our index, so should return empty results
        let query = Bm25Query::new("The running dogs".to_string(), user_id);
        let results = searcher.search(&query).unwrap();

        assert!(results.is_empty(), "Should return empty for unknown terms");
    }

    #[test]
    fn test_empty_query() {
        let tmp_dir = TempDir::new("test_bm25_empty").unwrap();
        let base_dir = tmp_dir.path().to_str().unwrap().to_string();
        let (user_id, collector, max_point_ids) = setup_test_data(&base_dir);

        let stats_writer = Bm25StatsWriter::new(base_dir.clone());
        stats_writer.write(&collector, &max_point_ids).unwrap();
        build_test_index(&base_dir, user_id);

        let config = BM25Config::default();
        let searcher =
            Bm25Searcher::new(base_dir.clone(), format!("{}/terms", base_dir), config).unwrap();

        let query = Bm25Query::new("".to_string(), user_id);
        let results = searcher.search(&query).unwrap();

        assert!(results.is_empty());
        assert_eq!(results.total_hits, 0);
    }

    #[test]
    fn test_query_only_stop_words() {
        let tmp_dir = TempDir::new("test_bm25_stop").unwrap();
        let base_dir = tmp_dir.path().to_str().unwrap().to_string();
        let (user_id, collector, max_point_ids) = setup_test_data(&base_dir);

        let stats_writer = Bm25StatsWriter::new(base_dir.clone());
        stats_writer.write(&collector, &max_point_ids).unwrap();
        build_test_index(&base_dir, user_id);

        let config = BM25Config::default();
        let searcher =
            Bm25Searcher::new(base_dir.clone(), format!("{}/terms", base_dir), config).unwrap();

        let query = Bm25Query::new("the and of".to_string(), user_id);
        let results = searcher.search(&query).unwrap();

        assert!(results.is_empty());
        assert_eq!(results.total_hits, 0);
    }

    #[test]
    fn test_unknown_user() {
        let tmp_dir = TempDir::new("test_bm25_user").unwrap();
        let base_dir = tmp_dir.path().to_str().unwrap().to_string();
        let (user_id, collector, max_point_ids) = setup_test_data(&base_dir);

        let stats_writer = Bm25StatsWriter::new(base_dir.clone());
        stats_writer.write(&collector, &max_point_ids).unwrap();
        build_test_index(&base_dir, user_id);

        let config = BM25Config::default();
        let searcher =
            Bm25Searcher::new(base_dir.clone(), format!("{}/terms", base_dir), config).unwrap();

        let unknown_user = 99999u128;
        let query = Bm25Query::new("date".to_string(), unknown_user);
        let result = searcher.search(&query);

        assert!(result.is_err());
    }

    #[test]
    fn test_unknown_term() {
        let tmp_dir = TempDir::new("test_bm25_term").unwrap();
        let base_dir = tmp_dir.path().to_str().unwrap().to_string();
        let (user_id, collector, max_point_ids) = setup_test_data(&base_dir);

        let stats_writer = Bm25StatsWriter::new(base_dir.clone());
        stats_writer.write(&collector, &max_point_ids).unwrap();
        build_test_index(&base_dir, user_id);

        let config = BM25Config::default();
        let searcher =
            Bm25Searcher::new(base_dir.clone(), format!("{}/terms", base_dir), config).unwrap();

        let query = Bm25Query::new("grape".to_string(), user_id);
        let results = searcher.search(&query).unwrap();

        assert!(results.is_empty());
        assert_eq!(results.total_hits, 0);
    }

    #[test]
    fn test_max_results_limiting() {
        let tmp_dir = TempDir::new("test_bm25_limit").unwrap();
        let base_dir = tmp_dir.path().to_str().unwrap().to_string();
        let (user_id, collector, max_point_ids) = setup_test_data(&base_dir);

        let stats_writer = Bm25StatsWriter::new(base_dir.clone());
        stats_writer.write(&collector, &max_point_ids).unwrap();
        build_test_index(&base_dir, user_id);

        let config = BM25Config::default();
        let searcher =
            Bm25Searcher::new(base_dir.clone(), format!("{}/terms", base_dir), config).unwrap();

        // "apple" appears in 3 docs (0, 1, 3), but with IDF floored at 0 for 60% DF
        // Let's use "date" which appears in 2 docs (3, 4)
        let query = Bm25Query::with_max_results("date".to_string(), user_id, 1);
        let results = searcher.search(&query).unwrap();

        assert_eq!(results.results.len(), 1, "Should return only 1 result");
        assert_eq!(results.total_hits, 2, "Should report 2 total hits");
    }
}
