/// Hybrid Search Query Types
///
/// Provides types for representing hybrid search queries that combine
/// both text (BM25) and vector similarity search modalities.

/// Query mode for hybrid search
///
/// Determines which search modalities to use when executing a query.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QueryMode {
    /// Search using only BM25 text relevance
    TextOnly,
    /// Search using only vector similarity
    VectorOnly,
    /// Search using both text and vector, combining scores
    Hybrid,
}

impl Default for QueryMode {
    fn default() -> Self {
        QueryMode::Hybrid
    }
}

/// A hybrid search query combining text and vector search
///
/// Represents a query that can search across both text (BM25) and vector
/// modalities, with configurable parameters for result ranking.
#[derive(Debug, Clone)]
pub struct HybridQuery {
    /// Text query for BM25 search (optional for vector-only mode)
    pub text_query: Option<String>,
    /// Vector query for similarity search (optional for text-only mode)
    pub vector_query: Option<Vec<f32>>,
    /// Which user's documents to search
    pub user_id: u128,
    /// Maximum number of results to return
    pub top_k: usize,
    /// Weight for BM25 scores in fusion (1.0 = text only, 0.0 = vector only, 0.5 = balanced)
    pub alpha: f32,
    /// Query mode determining which modalities to use
    pub mode: QueryMode,
}

impl HybridQuery {
    /// Create a new hybrid query with both text and vector components
    ///
    /// # Arguments
    /// * `text_query` - The text query for BM25 search
    /// * `vector_query` - The vector query for similarity search
    /// * `user_id` - The user whose documents to search
    /// * `top_k` - Maximum number of results to return
    ///
    /// # Examples
    /// ```
    /// use index::bm25::{HybridQuery, QueryMode};
    ///
    /// let query = HybridQuery::new(
    ///     "apple banana".to_string(),
    ///     vec![1.0, 2.0, 3.0, 4.0],
    ///     123u128,
    ///     10
    /// );
    /// assert_eq!(query.alpha, 0.5);
    /// assert_eq!(query.mode, QueryMode::Hybrid);
    /// ```
    pub fn new(text_query: String, vector_query: Vec<f32>, user_id: u128, top_k: usize) -> Self {
        Self {
            text_query: Some(text_query),
            vector_query: Some(vector_query),
            user_id,
            top_k,
            alpha: 0.5, // Default: balanced weighting
            mode: QueryMode::Hybrid,
        }
    }

    /// Create a new hybrid query with custom configuration
    ///
    /// # Arguments
    /// * `text_query` - The text query for BM25 search (optional)
    /// * `vector_query` - The vector query for similarity search (optional)
    /// * `user_id` - The user whose documents to search
    /// * `top_k` - Maximum number of results to return
    /// * `alpha` - Weight for BM25 scores [0.0, 1.0]
    /// * `mode` - Query mode
    ///
    /// # Examples
    /// ```
    /// use index::bm25::{HybridQuery, QueryMode};
    ///
    /// let query = HybridQuery::with_config(
    ///     Some("apple".to_string()),
    ///     Some(vec![1.0, 2.0, 3.0]),
    ///     123u128,
    ///     20,
    ///     0.7,
    ///     QueryMode::Hybrid,
    /// );
    /// assert_eq!(query.alpha, 0.7);
    /// assert_eq!(query.top_k, 20);
    /// ```
    pub fn with_config(
        text_query: Option<String>,
        vector_query: Option<Vec<f32>>,
        user_id: u128,
        top_k: usize,
        alpha: f32,
        mode: QueryMode,
    ) -> Self {
        let alpha = alpha.clamp(0.0, 1.0);
        Self {
            text_query,
            vector_query,
            user_id,
            top_k,
            alpha,
            mode,
        }
    }

    /// Create a text-only query
    ///
    /// # Arguments
    /// * `text` - The text query for BM25 search
    /// * `user_id` - The user whose documents to search
    /// * `top_k` - Maximum number of results to return
    pub fn text_only(text: String, user_id: u128, top_k: usize) -> Self {
        Self {
            text_query: Some(text),
            vector_query: None,
            user_id,
            top_k,
            alpha: 1.0, // Text only
            mode: QueryMode::TextOnly,
        }
    }

    /// Create a vector-only query
    ///
    /// # Arguments
    /// * `vector` - The vector query for similarity search
    /// * `user_id` - The user whose documents to search
    /// * `top_k` - Maximum number of results to return
    pub fn vector_only(vector: Vec<f32>, user_id: u128, top_k: usize) -> Self {
        Self {
            text_query: None,
            vector_query: Some(vector),
            user_id,
            top_k,
            alpha: 0.0, // Vector only
            mode: QueryMode::VectorOnly,
        }
    }

    /// Set the alpha value for score fusion
    ///
    /// # Arguments
    /// * `alpha` - Weight for BM25 scores [0.0, 1.0]
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha.clamp(0.0, 1.0);
        self
    }

    /// Set the query mode
    ///
    /// # Arguments
    /// * `mode` - The query mode to use
    pub fn with_mode(mut self, mode: QueryMode) -> Self {
        self.mode = mode;
        self
    }

    /// Check if the query should execute BM25 text search
    pub fn should_search_text(&self) -> bool {
        self.text_query.is_some()
            && (self.mode == QueryMode::TextOnly || self.mode == QueryMode::Hybrid)
    }

    /// Check if the query should execute vector search
    pub fn should_search_vector(&self) -> bool {
        self.vector_query.is_some()
            && (self.mode == QueryMode::VectorOnly || self.mode == QueryMode::Hybrid)
    }

    /// Get the text query reference
    pub fn text(&self) -> Option<&str> {
        self.text_query.as_deref()
    }

    /// Get the vector query reference
    pub fn vector(&self) -> Option<&[f32]> {
        self.vector_query.as_deref()
    }
}

impl Default for HybridQuery {
    fn default() -> Self {
        Self {
            text_query: None,
            vector_query: None,
            user_id: 0,
            top_k: 10,
            alpha: 0.5,
            mode: QueryMode::Hybrid,
        }
    }
}

/// A single hybrid search result
///
/// Contains the document ID along with its individual BM25 and vector scores,
/// plus the combined hybrid score used for final ranking.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HybridSearchResult {
    /// Document ID
    pub doc_id: u128,
    /// BM25 text relevance score (higher = more relevant)
    pub bm25_score: f32,
    /// Vector similarity score (typically higher = more similar)
    pub vector_score: f32,
    /// Combined hybrid score after fusion
    pub hybrid_score: f32,
}

impl HybridSearchResult {
    /// Create a new hybrid search result
    ///
    /// # Arguments
    /// * `doc_id` - Document ID
    /// * `bm25_score` - BM25 text relevance score
    /// * `vector_score` - Vector similarity score
    /// * `hybrid_score` - Combined hybrid score
    pub fn new(doc_id: u128, bm25_score: f32, vector_score: f32, hybrid_score: f32) -> Self {
        Self {
            doc_id,
            bm25_score,
            vector_score,
            hybrid_score,
        }
    }
}

impl Eq for HybridSearchResult {}

impl Ord for HybridSearchResult {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Compare by hybrid_score descending (higher scores first)
        // If scores are equal, compare by doc_id ascending for deterministic ordering
        match self.hybrid_score.partial_cmp(&other.hybrid_score) {
            Some(std::cmp::Ordering::Equal) => self.doc_id.cmp(&other.doc_id),
            Some(ord) => ord.reverse(), // Reverse for descending order
            None => std::cmp::Ordering::Equal, // NaN case
        }
    }
}

impl PartialOrd for HybridSearchResult {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Container for hybrid search results
///
/// Contains the ranked list of hybrid search results and metadata.
#[derive(Debug, Clone)]
pub struct HybridSearchResults {
    /// Results sorted by hybrid_score descending (highest relevance first)
    pub results: Vec<HybridSearchResult>,
    /// Total number of documents that matched either text or vector query
    pub total_hits: usize,
    /// Number of documents with both text and vector scores
    pub overlapping_hits: usize,
}

impl HybridSearchResults {
    /// Create new hybrid search results
    ///
    /// Results should already be sorted by hybrid_score descending.
    pub fn new(
        results: Vec<HybridSearchResult>,
        total_hits: usize,
        overlapping_hits: usize,
    ) -> Self {
        Self {
            results,
            total_hits,
            overlapping_hits,
        }
    }

    /// Create empty search results
    pub fn empty() -> Self {
        Self {
            results: Vec::new(),
            total_hits: 0,
            overlapping_hits: 0,
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
    pub fn top_result(&self) -> Option<&HybridSearchResult> {
        self.results.first()
    }

    /// Get the top N results
    pub fn top_n(&self, n: usize) -> &[HybridSearchResult] {
        &self.results[..self.results.len().min(n)]
    }

    /// Convert to (doc_id, hybrid_score) pairs
    pub fn to_doc_scores(&self) -> Vec<(u128, f32)> {
        self.results
            .iter()
            .map(|r| (r.doc_id, r.hybrid_score))
            .collect()
    }
}

impl Default for HybridSearchResults {
    fn default() -> Self {
        Self::empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test 1: HybridQuery::new creates query with default alpha=0.5, top_k=10
    #[test]
    fn test_hybrid_query_new_defaults() {
        let query = HybridQuery::new("test query".to_string(), vec![1.0, 2.0, 3.0], 123u128, 10);

        assert_eq!(query.alpha, 0.5);
        assert_eq!(query.top_k, 10);
        assert_eq!(query.user_id, 123);
        assert_eq!(query.mode, QueryMode::Hybrid);
        assert_eq!(query.text(), Some("test query"));
        assert!(query.vector().is_some());
    }

    // Test 2: HybridQuery::with_config sets custom alpha and top_k
    #[test]
    fn test_hybrid_query_with_config() {
        let query = HybridQuery::with_config(
            Some("apple".to_string()),
            Some(vec![1.0, 2.0, 3.0]),
            456u128,
            25,
            0.75,
            QueryMode::Hybrid,
        );

        assert_eq!(query.alpha, 0.75);
        assert_eq!(query.top_k, 25);
        assert_eq!(query.user_id, 456);
    }

    // Test 3: HybridSearchResult stores doc_id, bm25_score, vector_score, hybrid_score
    #[test]
    fn test_hybrid_search_result_fields() {
        let result = HybridSearchResult::new(42u128, 0.8, 0.6, 0.7);

        assert_eq!(result.doc_id, 42);
        assert_eq!(result.bm25_score, 0.8);
        assert_eq!(result.vector_score, 0.6);
        assert_eq!(result.hybrid_score, 0.7);
    }

    // Test 4: Results implement Ord for sorting by hybrid_score descending
    #[test]
    fn test_hybrid_result_ordering() {
        let low = HybridSearchResult::new(1u128, 0.5, 0.5, 0.3);
        let high = HybridSearchResult::new(2u128, 0.8, 0.8, 0.9);
        let medium = HybridSearchResult::new(3u128, 0.6, 0.6, 0.6);

        // For descending sort: high should come first (high < low in Ord terms)
        assert!(high < low, "Higher score should come before lower");
        assert!(medium < low, "Medium should come before low");
        assert!(high < medium, "High should come before medium");
    }

    // Test 5: QueryMode::TextOnly, VectorOnly, Hybrid variants work correctly
    #[test]
    fn test_query_modes() {
        let text_only = HybridQuery::text_only("query".to_string(), 1u128, 10);
        assert_eq!(text_only.mode, QueryMode::TextOnly);
        assert_eq!(text_only.alpha, 1.0);
        assert!(text_only.should_search_text());
        assert!(!text_only.should_search_vector());

        let vector_only = HybridQuery::vector_only(vec![1.0, 2.0], 1u128, 10);
        assert_eq!(vector_only.mode, QueryMode::VectorOnly);
        assert_eq!(vector_only.alpha, 0.0);
        assert!(!vector_only.should_search_text());
        assert!(vector_only.should_search_vector());

        let hybrid = HybridQuery::new("text".to_string(), vec![1.0], 1u128, 10);
        assert_eq!(hybrid.mode, QueryMode::Hybrid);
        assert!(hybrid.should_search_text());
        assert!(hybrid.should_search_vector());
    }

    // Test: Alpha clamping
    #[test]
    fn test_alpha_clamping() {
        let query_high = HybridQuery::with_config(
            Some("test".to_string()),
            Some(vec![1.0]),
            1u128,
            10,
            1.5, // Should be clamped to 1.0
            QueryMode::Hybrid,
        );
        assert_eq!(query_high.alpha, 1.0);

        let query_low = HybridQuery::with_config(
            Some("test".to_string()),
            Some(vec![1.0]),
            1u128,
            10,
            -0.5, // Should be clamped to 0.0
            QueryMode::Hybrid,
        );
        assert_eq!(query_low.alpha, 0.0);
    }

    // Test: HybridSearchResults container
    #[test]
    fn test_hybrid_search_results() {
        let results = HybridSearchResults::new(
            vec![
                HybridSearchResult::new(1u128, 0.8, 0.6, 0.7),
                HybridSearchResult::new(2u128, 0.5, 0.9, 0.7),
            ],
            5, // total_hits
            2, // overlapping_hits
        );

        assert_eq!(results.len(), 2);
        assert!(!results.is_empty());
        assert_eq!(results.total_hits, 5);
        assert_eq!(results.overlapping_hits, 2);
    }

    // Test: HybridSearchResults::to_doc_scores
    #[test]
    fn test_to_doc_scores() {
        let results = HybridSearchResults::new(
            vec![
                HybridSearchResult::new(1u128, 0.8, 0.6, 0.9),
                HybridSearchResult::new(2u128, 0.5, 0.7, 0.6),
            ],
            2,
            2,
        );

        let doc_scores = results.to_doc_scores();
        assert_eq!(doc_scores, vec![(1u128, 0.9), (2u128, 0.6)]);
    }

    // Test: HybridSearchResult tie-breaking by doc_id
    #[test]
    fn test_hybrid_result_tie_breaking() {
        let a = HybridSearchResult::new(1u128, 0.5, 0.5, 0.7);
        let b = HybridSearchResult::new(2u128, 0.6, 0.4, 0.7);

        // Same score, lower doc_id should come first
        assert!(
            a < b,
            "Lower doc_id should come first when scores are equal"
        );
    }

    // Test: Default implementations
    #[test]
    fn test_defaults() {
        let query: HybridQuery = Default::default();
        assert_eq!(query.alpha, 0.5);
        assert_eq!(query.top_k, 10);
        assert_eq!(query.mode, QueryMode::Hybrid);

        let results: HybridSearchResults = Default::default();
        assert!(results.is_empty());
        assert_eq!(results.total_hits, 0);

        let mode: QueryMode = Default::default();
        assert_eq!(mode, QueryMode::Hybrid);
    }

    // Test: HybridSearchResults empty
    #[test]
    fn test_empty_results() {
        let results = HybridSearchResults::empty();
        assert!(results.is_empty());
        assert_eq!(results.len(), 0);
        assert!(results.top_result().is_none());
    }

    // Test: HybridSearchResults top_n
    #[test]
    fn test_top_n() {
        let results = HybridSearchResults::new(
            vec![
                HybridSearchResult::new(1u128, 0.9, 0.9, 0.9),
                HybridSearchResult::new(2u128, 0.7, 0.7, 0.7),
                HybridSearchResult::new(3u128, 0.5, 0.5, 0.5),
                HybridSearchResult::new(4u128, 0.3, 0.3, 0.3),
            ],
            4,
            4,
        );

        let top_2 = results.top_n(2);
        assert_eq!(top_2.len(), 2);
        assert_eq!(top_2[0].doc_id, 1);
        assert_eq!(top_2[1].doc_id, 2);

        // Asking for more than available
        let top_10 = results.top_n(10);
        assert_eq!(top_10.len(), 4);
    }

    // Test: should_search_text and should_search_vector
    #[test]
    fn test_should_search_methods() {
        // Text-only with text present
        let text_only = HybridQuery::text_only("query".to_string(), 1u128, 10);
        assert!(text_only.should_search_text());
        assert!(!text_only.should_search_vector());

        // Vector-only with vector present
        let vector_only = HybridQuery::vector_only(vec![1.0], 1u128, 10);
        assert!(!vector_only.should_search_text());
        assert!(vector_only.should_search_vector());

        // Hybrid with both present
        let hybrid = HybridQuery::new("text".to_string(), vec![1.0], 1u128, 10);
        assert!(hybrid.should_search_text());
        assert!(hybrid.should_search_vector());

        // Hybrid with only text
        let hybrid_text = HybridQuery::with_config(
            Some("text".to_string()),
            None,
            1u128,
            10,
            0.5,
            QueryMode::Hybrid,
        );
        assert!(hybrid_text.should_search_text());
        assert!(!hybrid_text.should_search_vector());

        // Hybrid with only vector
        let hybrid_vector =
            HybridQuery::with_config(None, Some(vec![1.0]), 1u128, 10, 0.5, QueryMode::Hybrid);
        assert!(!hybrid_vector.should_search_text());
        assert!(hybrid_vector.should_search_vector());
    }

    // Test: with_alpha builder
    #[test]
    fn test_with_alpha() {
        let query = HybridQuery::new("text".to_string(), vec![1.0], 1u128, 10).with_alpha(0.8);
        assert_eq!(query.alpha, 0.8);
    }

    // Test: with_mode builder
    #[test]
    fn test_with_mode() {
        let query = HybridQuery::new("text".to_string(), vec![1.0], 1u128, 10)
            .with_mode(QueryMode::TextOnly);
        assert_eq!(query.mode, QueryMode::TextOnly);
    }
}
