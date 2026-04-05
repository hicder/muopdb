/// Hybrid Score Fusion for combining BM25 and Vector Search Results
///
/// Provides algorithms for fusing text relevance scores (BM25) with vector
/// similarity scores using linear combination and other fusion techniques.
///
/// The core formula is:
/// hybrid_score = alpha * bm25_score + (1 - alpha) * vector_score
///
/// Where:
/// - alpha ∈ [0, 1] controls the weighting (1.0 = text only, 0.0 = vector only)
/// - bm25_score is normalized to [0, 1] range
/// - vector_score is typically in [0, 1] or [-1, 1] range
///
/// A hybrid search candidate with both text and vector scores
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HybridCandidate {
    /// Document ID (u128 for multi-user support)
    pub doc_id: u128,
    /// Normalized BM25 text relevance score [0, 1]
    pub bm25_score: f32,
    /// Vector similarity score [0, 1] or [-1, 1]
    pub vector_score: f32,
    /// Combined hybrid score after fusion
    pub hybrid_score: f32,
}

impl HybridCandidate {
    /// Create a new hybrid candidate
    ///
    /// # Arguments
    /// * `doc_id` - Document ID
    /// * `bm25_score` - BM25 score (will be used as-is, should be pre-normalized)
    /// * `vector_score` - Vector similarity score
    /// * `hybrid_score` - Combined score (computed by fusion)
    pub fn new(doc_id: u128, bm25_score: f32, vector_score: f32, hybrid_score: f32) -> Self {
        Self {
            doc_id,
            bm25_score,
            vector_score,
            hybrid_score,
        }
    }

    /// Create a new candidate with only BM25 score (vector score = 0)
    pub fn with_bm25_only(doc_id: u128, bm25_score: f32, alpha: f32) -> Self {
        let hybrid_score = alpha * bm25_score; // vector_score = 0
        Self {
            doc_id,
            bm25_score,
            vector_score: 0.0,
            hybrid_score,
        }
    }

    /// Create a new candidate with only vector score (BM25 score = 0)
    pub fn with_vector_only(doc_id: u128, vector_score: f32, alpha: f32) -> Self {
        let hybrid_score = (1.0 - alpha) * vector_score; // bm25_score = 0
        Self {
            doc_id,
            bm25_score: 0.0,
            vector_score,
            hybrid_score,
        }
    }
}

impl Eq for HybridCandidate {}

impl Ord for HybridCandidate {
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

impl PartialOrd for HybridCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Configuration for score fusion
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FusionConfig {
    /// Weight for BM25 scores (alpha), vector weight is (1 - alpha)
    /// Range: [0.0, 1.0]
    /// - 1.0 = text only (BM25)
    /// - 0.5 = equal weighting
    /// - 0.0 = vector only
    pub alpha: f32,
}

impl FusionConfig {
    /// Create a new fusion config with the given alpha
    ///
    /// # Arguments
    /// * `alpha` - Weight for BM25 scores [0.0, 1.0]
    ///
    /// # Panics
    /// Panics in debug mode if alpha is outside [0, 1]
    pub fn new(alpha: f32) -> Self {
        debug_assert!(
            (0.0..=1.0).contains(&alpha),
            "Alpha must be in [0.0, 1.0], got {}",
            alpha
        );
        Self {
            alpha: alpha.clamp(0.0, 1.0),
        }
    }

    /// Create a config with equal weighting (alpha = 0.5)
    pub fn balanced() -> Self {
        Self { alpha: 0.5 }
    }

    /// Create a text-only config (alpha = 1.0)
    pub fn text_only() -> Self {
        Self { alpha: 1.0 }
    }

    /// Create a vector-only config (alpha = 0.0)
    pub fn vector_only() -> Self {
        Self { alpha: 0.0 }
    }
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self::balanced()
    }
}

/// Hybrid Score Fusion implementation
///
/// Combines BM25 text relevance scores with vector similarity scores
/// using linear combination fusion with configurable weighting.
///
/// # Example
/// ```
/// use index::bm25::{HybridScoreFusion, FusionConfig};
///
/// let fusion = HybridScoreFusion::new(FusionConfig::balanced());
/// let hybrid_score = fusion.combine(0.8, 0.6); // bm25=0.8, vector=0.6
/// assert!((hybrid_score - 0.7).abs() < 0.001); // 0.5 * 0.8 + 0.5 * 0.6 = 0.7
/// ```
#[derive(Debug, Clone)]
pub struct HybridScoreFusion {
    config: FusionConfig,
}

impl HybridScoreFusion {
    /// Create a new hybrid score fusion with the given configuration
    pub fn new(config: FusionConfig) -> Self {
        Self { config }
    }

    /// Create a new fusion with the specified alpha value
    ///
    /// # Arguments
    /// * `alpha` - Weight for BM25 scores [0.0, 1.0]
    pub fn with_alpha(alpha: f32) -> Self {
        Self::new(FusionConfig::new(alpha))
    }

    /// Get the current fusion configuration
    pub fn config(&self) -> &FusionConfig {
        &self.config
    }

    /// Compute the hybrid score using linear combination
    ///
    /// Formula: alpha * bm25_score + (1 - alpha) * vector_score
    ///
    /// # Arguments
    /// * `bm25_score` - BM25 text relevance score (should be normalized to [0, 1])
    /// * `vector_score` - Vector similarity score (typically in [0, 1])
    ///
    /// # Returns
    /// Combined hybrid score
    pub fn combine(&self, bm25_score: f32, vector_score: f32) -> f32 {
        let alpha = self.config.alpha;
        alpha * bm25_score + (1.0 - alpha) * vector_score
    }

    /// Compute hybrid score with explicit alpha (for per-query customization)
    ///
    /// # Arguments
    /// * `bm25_score` - BM25 text relevance score
    /// * `vector_score` - Vector similarity score
    /// * `alpha` - Weight for BM25 (clamped to [0, 1])
    ///
    /// # Returns
    /// Combined hybrid score using the provided alpha
    pub fn weighted_combine(&self, bm25_score: f32, vector_score: f32, alpha: f32) -> f32 {
        let alpha = alpha.clamp(0.0, 1.0);
        alpha * bm25_score + (1.0 - alpha) * vector_score
    }

    /// Merge BM25 and vector search results by doc_id
    ///
    /// Combines results from both search types, creating HybridCandidates
    /// for each document. Documents present in only one source get 0.0 for
    /// the missing score. Entries with NaN scores are filtered out.
    ///
    /// # Arguments
    /// * `bm25_results` - Vector of (doc_id, bm25_score) pairs
    /// * `vector_results` - Vector of (doc_id, vector_score) pairs
    ///
    /// # Returns
    /// Vector of HybridCandidates sorted by hybrid score descending
    pub fn merge_results(
        &self,
        bm25_results: &[(u128, f32)],
        vector_results: &[(u128, f32)],
    ) -> Vec<HybridCandidate> {
        use std::collections::HashMap;

        // Build lookup maps, filtering out NaN values
        let mut bm25_map: HashMap<u128, f32> = HashMap::new();
        for &(doc_id, score) in bm25_results {
            if score.is_finite() {
                bm25_map.insert(doc_id, score);
            }
        }

        let mut vector_map: HashMap<u128, f32> = HashMap::new();
        for &(doc_id, score) in vector_results {
            if score.is_finite() {
                vector_map.insert(doc_id, score);
            }
        }

        // Collect all unique doc_ids
        let all_doc_ids: std::collections::HashSet<u128> =
            bm25_map.keys().chain(vector_map.keys()).copied().collect();

        // Create hybrid candidates
        let mut candidates: Vec<HybridCandidate> = all_doc_ids
            .iter()
            .map(|&doc_id| {
                let bm25_score = *bm25_map.get(&doc_id).unwrap_or(&0.0);
                let vector_score = *vector_map.get(&doc_id).unwrap_or(&0.0);
                let hybrid_score = self.combine(bm25_score, vector_score);

                HybridCandidate::new(doc_id, bm25_score, vector_score, hybrid_score)
            })
            .collect();

        // Sort by hybrid score descending
        candidates.sort();

        candidates
    }

    /// Create hybrid candidates from pre-merged data
    ///
    /// Takes a list of doc_ids with both scores already aligned,
    /// computes hybrid scores, and returns sorted results.
    ///
    /// # Arguments
    /// * `docs` - Vector of (doc_id, bm25_score, vector_score) tuples
    ///
    /// # Returns
    /// Vector of HybridCandidates sorted by hybrid score descending
    pub fn fuse_aligned_results(&self, docs: &[(u128, f32, f32)]) -> Vec<HybridCandidate> {
        let mut candidates: Vec<HybridCandidate> = docs
            .iter()
            .map(|&(doc_id, bm25_score, vector_score)| {
                let hybrid_score = self.combine(bm25_score, vector_score);
                HybridCandidate::new(doc_id, bm25_score, vector_score, hybrid_score)
            })
            .collect();

        candidates.sort();
        candidates
    }
}

impl Default for HybridScoreFusion {
    fn default() -> Self {
        Self::new(FusionConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test 1: Equal weights (alpha=0.5) -> average
    #[test]
    fn test_combine_equal_weights() {
        let fusion = HybridScoreFusion::with_alpha(0.5);
        let hybrid = fusion.combine(0.8, 0.6);
        assert!(
            (hybrid - 0.7).abs() < 0.0001,
            "Expected 0.7, got {}",
            hybrid
        );
    }

    // Test 2: Text-only (alpha=1.0) -> bm25 only
    #[test]
    fn test_combine_text_only() {
        let fusion = HybridScoreFusion::with_alpha(1.0);
        let hybrid = fusion.combine(0.8, 0.6);
        assert!(
            (hybrid - 0.8).abs() < 0.0001,
            "Expected 0.8, got {}",
            hybrid
        );
    }

    // Test 3: Vector-only (alpha=0.0) -> vector only
    #[test]
    fn test_combine_vector_only() {
        let fusion = HybridScoreFusion::with_alpha(0.0);
        let hybrid = fusion.combine(0.8, 0.6);
        assert!(
            (hybrid - 0.6).abs() < 0.0001,
            "Expected 0.6, got {}",
            hybrid
        );
    }

    // Test 4: Zero BM25 score
    #[test]
    fn test_combine_zero_bm25() {
        let fusion = HybridScoreFusion::with_alpha(0.3);
        let hybrid = fusion.combine(0.0, 1.0);
        // 0.3 * 0.0 + 0.7 * 1.0 = 0.7
        assert!(
            (hybrid - 0.7).abs() < 0.0001,
            "Expected 0.7, got {}",
            hybrid
        );
    }

    // Test 5: Alpha clamped to [0, 1]
    #[test]
    fn test_alpha_clamped() {
        // Alpha > 1 should be clamped to 1
        let fusion_high = HybridScoreFusion::with_alpha(1.5);
        assert_eq!(fusion_high.config.alpha, 1.0);

        // Alpha < 0 should be clamped to 0
        let fusion_low = HybridScoreFusion::with_alpha(-0.2);
        assert_eq!(fusion_low.config.alpha, 0.0);

        // Verify clamping affects results
        let hybrid_high = fusion_high.combine(0.5, 0.9);
        assert!((hybrid_high - 0.5).abs() < 0.0001); // text-only

        let hybrid_low = fusion_low.combine(0.5, 0.9);
        assert!((hybrid_low - 0.9).abs() < 0.0001); // vector-only
    }

    // Test 6: Merge results from both sources
    #[test]
    fn test_merge_results() {
        let fusion = HybridScoreFusion::with_alpha(0.5);

        // BM25 results: doc 1 has high score, doc 2 has low score
        let bm25_results = vec![(1u128, 0.8f32), (2u128, 0.3f32)];

        // Vector results: doc 2 has high score, doc 3 has medium score
        let vector_results = vec![(2u128, 0.9f32), (3u128, 0.5f32)];

        let merged = fusion.merge_results(&bm25_results, &vector_results);

        // Should have 3 unique doc_ids
        assert_eq!(merged.len(), 3);

        // Doc 2 has both scores: (0.3 + 0.9) / 2 = 0.6
        let doc2 = merged.iter().find(|c| c.doc_id == 2).unwrap();
        assert!((doc2.hybrid_score - 0.6).abs() < 0.0001);

        // Doc 1 has only BM25: 0.8 * 0.5 = 0.4
        let doc1 = merged.iter().find(|c| c.doc_id == 1).unwrap();
        assert!((doc1.hybrid_score - 0.4).abs() < 0.0001);
        assert_eq!(doc1.vector_score, 0.0);

        // Doc 3 has only vector: 0.5 * 0.5 = 0.25
        let doc3 = merged.iter().find(|c| c.doc_id == 3).unwrap();
        assert!((doc3.hybrid_score - 0.25).abs() < 0.0001);
        assert_eq!(doc3.bm25_score, 0.0);

        // Results should be sorted by hybrid score descending
        assert_eq!(merged[0].doc_id, 2); // highest score
    }

    // Test: FusionConfig creation
    #[test]
    fn test_fusion_config_new() {
        let config = FusionConfig::new(0.3);
        assert_eq!(config.alpha, 0.3);
    }

    // Test: FusionConfig with clamping
    #[test]
    fn test_fusion_config_clamping() {
        let config = FusionConfig::new(1.5);
        assert_eq!(config.alpha, 1.0);

        let config = FusionConfig::new(-0.5);
        assert_eq!(config.alpha, 0.0);
    }

    // Test: FusionConfig factory methods
    #[test]
    fn test_fusion_config_factories() {
        assert_eq!(FusionConfig::balanced().alpha, 0.5);
        assert_eq!(FusionConfig::text_only().alpha, 1.0);
        assert_eq!(FusionConfig::vector_only().alpha, 0.0);
    }

    // Test: FusionConfig default
    #[test]
    fn test_fusion_config_default() {
        let config: FusionConfig = Default::default();
        assert_eq!(config.alpha, 0.5);
    }

    // Test: HybridScoreFusion default
    #[test]
    fn test_hybrid_fusion_default() {
        let fusion: HybridScoreFusion = Default::default();
        assert_eq!(fusion.config.alpha, 0.5);
    }

    // Test: HybridCandidate creation
    #[test]
    fn test_hybrid_candidate_new() {
        let candidate = HybridCandidate::new(42u128, 0.8, 0.6, 0.7);
        assert_eq!(candidate.doc_id, 42);
        assert_eq!(candidate.bm25_score, 0.8);
        assert_eq!(candidate.vector_score, 0.6);
        assert_eq!(candidate.hybrid_score, 0.7);
    }

    // Test: HybridCandidate BM25 only
    #[test]
    fn test_hybrid_candidate_bm25_only() {
        let candidate = HybridCandidate::with_bm25_only(1u128, 0.8, 0.5);
        assert_eq!(candidate.doc_id, 1);
        assert_eq!(candidate.bm25_score, 0.8);
        assert_eq!(candidate.vector_score, 0.0);
        // hybrid = 0.5 * 0.8 + 0.5 * 0.0 = 0.4
        assert!((candidate.hybrid_score - 0.4).abs() < 0.0001);
    }

    // Test: HybridCandidate vector only
    #[test]
    fn test_hybrid_candidate_vector_only() {
        let candidate = HybridCandidate::with_vector_only(1u128, 0.6, 0.3);
        assert_eq!(candidate.doc_id, 1);
        assert_eq!(candidate.bm25_score, 0.0);
        assert_eq!(candidate.vector_score, 0.6);
        // hybrid = 0.3 * 0.0 + 0.7 * 0.6 = 0.42
        assert!((candidate.hybrid_score - 0.42).abs() < 0.0001);
    }

    // Test: HybridCandidate ordering (sorted by score descending)
    #[test]
    fn test_hybrid_candidate_ordering() {
        let low = HybridCandidate::new(1u128, 0.0, 0.0, 0.3);
        let high = HybridCandidate::new(2u128, 0.0, 0.0, 0.8);
        let medium = HybridCandidate::new(3u128, 0.0, 0.0, 0.5);

        // High should come first in sorted order (low < high in Ord terms for descending sort)
        assert!(high < low, "Higher score should come first");
        assert!(medium < low, "Medium should come before low");
        assert!(high < medium, "High should come before medium");
    }

    // Test: HybridCandidate tie-breaking by doc_id
    #[test]
    fn test_hybrid_candidate_tie_breaking() {
        let a = HybridCandidate::new(1u128, 0.5, 0.5, 0.7);
        let b = HybridCandidate::new(2u128, 0.6, 0.4, 0.7);

        // Same score, lower doc_id should come first
        assert!(
            a < b,
            "Lower doc_id should come first when scores are equal"
        );
    }

    // Test: Merge empty results
    #[test]
    fn test_merge_empty_results() {
        let fusion = HybridScoreFusion::with_alpha(0.5);
        let merged = fusion.merge_results(&[], &[]);
        assert!(merged.is_empty());
    }

    // Test: Merge with only BM25 results
    #[test]
    fn test_merge_bm25_only() {
        let fusion = HybridScoreFusion::with_alpha(0.5);
        let bm25_results = vec![(1u128, 0.8), (2u128, 0.6)];
        let vector_results = vec![];

        let merged = fusion.merge_results(&bm25_results, &vector_results);
        assert_eq!(merged.len(), 2);

        let doc1 = merged.iter().find(|c| c.doc_id == 1).unwrap();
        assert!((doc1.hybrid_score - 0.4).abs() < 0.0001); // 0.5 * 0.8
        assert_eq!(doc1.vector_score, 0.0);
    }

    // Test: Merge with only vector results
    #[test]
    fn test_merge_vector_only() {
        let fusion = HybridScoreFusion::with_alpha(0.5);
        let bm25_results = vec![];
        let vector_results = vec![(1u128, 0.8), (2u128, 0.6)];

        let merged = fusion.merge_results(&bm25_results, &vector_results);
        assert_eq!(merged.len(), 2);

        let doc1 = merged.iter().find(|c| c.doc_id == 1).unwrap();
        assert!((doc1.hybrid_score - 0.4).abs() < 0.0001); // 0.5 * 0.8
        assert_eq!(doc1.bm25_score, 0.0);
    }

    // Test: Weighted combine with different alpha
    #[test]
    fn test_weighted_combine() {
        let fusion = HybridScoreFusion::with_alpha(0.5);
        // Use different alpha than the config
        let hybrid = fusion.weighted_combine(0.8, 0.6, 0.25);
        // 0.25 * 0.8 + 0.75 * 0.6 = 0.2 + 0.45 = 0.65
        assert!((hybrid - 0.65).abs() < 0.0001);
    }

    // Test: Fuse aligned results
    #[test]
    fn test_fuse_aligned_results() {
        let fusion = HybridScoreFusion::with_alpha(0.5);
        let docs = vec![
            (1u128, 0.8, 0.4), // doc_id, bm25, vector
            (2u128, 0.3, 0.9),
            (3u128, 0.6, 0.6),
        ];

        let fused = fusion.fuse_aligned_results(&docs);
        assert_eq!(fused.len(), 3);

        // Scores: doc1=0.6, doc2=0.6, doc3=0.6
        // All equal, tie-break by doc_id
        assert_eq!(fused[0].doc_id, 1);
        assert_eq!(fused[1].doc_id, 2);
        assert_eq!(fused[2].doc_id, 3);
    }

    // Test: Verify all scores in merged results are finite
    #[test]
    fn test_merged_results_finite() {
        let fusion = HybridScoreFusion::with_alpha(0.5);
        let bm25_results = vec![(1u128, 0.8), (2u128, f32::NAN)];
        let vector_results = vec![(1u128, 0.6), (3u128, 0.4)];

        let merged = fusion.merge_results(&bm25_results, &vector_results);

        for candidate in &merged {
            assert!(
                candidate.hybrid_score.is_finite(),
                "Hybrid score for doc {} should be finite",
                candidate.doc_id
            );
        }
    }
}
