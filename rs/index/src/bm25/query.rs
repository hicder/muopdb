/// BM25 Search Query Types
///
/// Provides types for representing BM25 text search queries and their results.

/// A BM25 search query
///
/// Represents a text search query with parameters for controlling
/// which user's documents to search and how many results to return.
#[derive(Debug, Clone)]
pub struct Bm25Query {
    /// The raw query text to search for
    pub text: String,
    /// Which user's documents to search
    pub user_id: u128,
    /// Maximum number of documents to return
    pub max_results: usize,
}

impl Bm25Query {
    /// Create a new BM25 query with default max_results
    ///
    /// # Arguments
    /// * `text` - The query text to search for
    /// * `user_id` - The user whose documents to search
    pub fn new(text: String, user_id: u128) -> Self {
        Self {
            text,
            user_id,
            max_results: 10, // Default
        }
    }

    /// Create a new BM25 query with explicit max_results
    ///
    /// # Arguments
    /// * `text` - The query text to search for
    /// * `user_id` - The user whose documents to search
    /// * `max_results` - Maximum number of results to return
    pub fn with_max_results(text: String, user_id: u128, max_results: usize) -> Self {
        Self {
            text,
            user_id,
            max_results,
        }
    }
}

impl Default for Bm25Query {
    fn default() -> Self {
        Self {
            text: String::new(),
            user_id: 0,
            max_results: 10,
        }
    }
}

/// A single search result with document ID and BM25 score
///
/// Contains the document ID and its BM25 relevance score.
/// Results are ordered by score in descending order (highest relevance first).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bm25SearchResult {
    /// The point ID (internal document ID)
    pub point_id: u32,
    /// The BM25 relevance score (higher = more relevant)
    pub score: f32,
}

impl Bm25SearchResult {
    /// Create a new search result
    pub fn new(point_id: u32, score: f32) -> Self {
        Self { point_id, score }
    }
}

impl Eq for Bm25SearchResult {}

impl Ord for Bm25SearchResult {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Compare by score descending (higher scores first)
        // If scores are equal, compare by point_id ascending for deterministic ordering
        match self.score.partial_cmp(&other.score) {
            Some(std::cmp::Ordering::Equal) => self.point_id.cmp(&other.point_id),
            Some(ord) => ord.reverse(), // Reverse for descending order
            None => std::cmp::Ordering::Equal, // NaN case
        }
    }
}

impl PartialOrd for Bm25SearchResult {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Container for BM25 search results
///
/// Contains the ranked list of search results and metadata about the search.
#[derive(Debug, Clone)]
pub struct Bm25SearchResults {
    /// Results sorted by score descending (highest relevance first)
    pub results: Vec<Bm25SearchResult>,
    /// Total number of documents that matched the query
    pub total_hits: usize,
}

impl Bm25SearchResults {
    /// Create new search results from a vector of results
    ///
    /// Results will be sorted by score descending.
    pub fn new(mut results: Vec<Bm25SearchResult>) -> Self {
        results.sort();
        let total_hits = results.len();
        Self {
            results: results.clone(),
            total_hits,
        }
    }

    /// Create new search results with explicit total_matches
    pub fn with_total_matches(results: Vec<Bm25SearchResult>, total_matches: usize) -> Self {
        Self {
            results,
            total_hits: total_matches,
        }
    }

    /// Create empty search results
    pub fn empty() -> Self {
        Self {
            results: Vec::new(),
            total_hits: 0,
        }
    }

    /// Get the number of results
    pub fn len(&self) -> usize {
        self.results.len()
    }

    /// Check if results are empty
    pub fn is_empty(&self) -> bool {
        self.results.is_empty()
    }

    /// Get the top result
    pub fn top_result(&self) -> Option<&Bm25SearchResult> {
        self.results.first()
    }

    /// Get the top N results
    pub fn top_n(&self, n: usize) -> &[Bm25SearchResult] {
        &self.results[..self.results.len().min(n)]
    }
}

impl Default for Bm25SearchResults {
    fn default() -> Self {
        Self::empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bm25_query_default() {
        let query = Bm25Query::default();
        assert_eq!(query.text, "");
        assert_eq!(query.user_id, 0);
        assert_eq!(query.max_results, 10);
    }

    #[test]
    fn test_bm25_query_new() {
        let query = Bm25Query::new("hello world".to_string(), 123u128);
        assert_eq!(query.text, "hello world");
        assert_eq!(query.user_id, 123);
        assert_eq!(query.max_results, 10);
    }

    #[test]
    fn test_bm25_query_with_max_results() {
        let query = Bm25Query::with_max_results("test".to_string(), 1u128, 50);
        assert_eq!(query.max_results, 50);
    }

    #[test]
    fn test_search_result_ordering() {
        // Higher scores should come first in sorting (descending order)
        // Note: In Rust, a < b means a comes before b in sorted order
        // So for descending scores, we want high < medium < low
        let high = Bm25SearchResult::new(1, 10.0);
        let medium = Bm25SearchResult::new(2, 5.0);
        let low = Bm25SearchResult::new(3, 1.0);

        // For descending sort order: high comes first, so high < medium
        assert!(high < medium, "High score should come before medium");
        assert!(medium < low, "Medium score should come before low");
        assert!(high < low, "High score should come before low");

        // Test tie-breaking: same score, lower point_id comes first
        let tie_low_id = Bm25SearchResult::new(1, 5.0);
        let tie_high_id = Bm25SearchResult::new(2, 5.0);
        assert!(
            tie_low_id < tie_high_id,
            "Lower point_id should come first when scores are equal"
        );
    }

    #[test]
    fn test_search_result_sorting() {
        let mut results = vec![
            Bm25SearchResult::new(1, 5.0),
            Bm25SearchResult::new(2, 10.0),
            Bm25SearchResult::new(3, 1.0),
        ];

        results.sort();

        // After sorting, should be in descending order by score
        assert_eq!(results[0].point_id, 2); // score 10.0
        assert_eq!(results[1].point_id, 1); // score 5.0
        assert_eq!(results[2].point_id, 3); // score 1.0
    }

    #[test]
    fn test_search_results_container() {
        let results = Bm25SearchResults::with_total_matches(
            vec![
                Bm25SearchResult::new(1, 10.0),
                Bm25SearchResult::new(2, 5.0),
            ],
            10, // total hits
        );

        assert_eq!(results.len(), 2);
        assert!(!results.is_empty());
        assert_eq!(results.total_hits, 10);
        assert_eq!(results.top_result().unwrap().point_id, 1);
    }

    #[test]
    fn test_empty_search_results() {
        let results = Bm25SearchResults::empty();
        assert!(results.is_empty());
        assert_eq!(results.len(), 0);
        assert!(results.top_result().is_none());
    }

    #[test]
    fn test_search_results_top_n() {
        let results = Bm25SearchResults::with_total_matches(
            vec![
                Bm25SearchResult::new(1, 10.0),
                Bm25SearchResult::new(2, 5.0),
                Bm25SearchResult::new(3, 3.0),
                Bm25SearchResult::new(4, 1.0),
            ],
            4,
        );

        let top_2 = results.top_n(2);
        assert_eq!(top_2.len(), 2);
        assert_eq!(top_2[0].point_id, 1);
        assert_eq!(top_2[1].point_id, 2);

        // Asking for more than available
        let top_10 = results.top_n(10);
        assert_eq!(top_10.len(), 4);
    }
}
