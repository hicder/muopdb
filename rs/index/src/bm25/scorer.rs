use tracing::debug;

use crate::bm25::config::BM25Config;

/// BM25 Scorer for computing relevance scores
///
/// BM25 (Best Matching 25) is a probabilistic ranking function that improves
/// upon TF-IDF by introducing term frequency saturation and document length normalization.
///
/// # Formula
/// Score = IDF × TF_component
///
/// ## IDF (Inverse Document Frequency)
/// IDF = ln((N - n + 0.5) / (n + 0.5))
/// Floored at 0 to prevent negative scores for common terms (>50% of docs)
///
/// ## TF Component (Term Frequency with Saturation)
/// TF = (f × (k1 + 1)) / (f + k1 × (1 - b + b × (dl / avgdl)))
///
/// Where:
/// - N = total number of documents
/// - n = document frequency (docs containing the term)
/// - f = term frequency in document
/// - dl = document length (number of terms)
/// - avgdl = average document length
/// - k1 = term frequency saturation parameter (default 1.2)
/// - b = length normalization parameter (default 0.75)
///
/// # Example
/// ```
/// use index::bm25::{BM25Config, BM25Scorer};
///
/// let config = BM25Config::default();
/// let scorer = BM25Scorer::new(config);
///
/// // Compute score for a term-document pair
/// let score = scorer.score(
///     3,           // term frequency
///     100,         // document length
///     80.0,        // average document length
///     10,          // document frequency (docs with this term)
///     1000,        // total documents
/// );
/// ```
pub struct BM25Scorer {
    config: BM25Config,
}

impl BM25Scorer {
    /// Create a new BM25Scorer with the given configuration
    pub fn new(config: BM25Config) -> Self {
        Self { config }
    }

    /// Get the configuration used by this scorer
    pub fn config(&self) -> &BM25Config {
        &self.config
    }

    /// Compute IDF with flooring at 0
    ///
    /// Formula: ln((N - n + 0.5) / (n + 0.5))
    /// Floored at 0 to prevent negative scores for high-frequency terms.
    ///
    /// Where:
    /// - N = total number of documents
    /// - n = document frequency (number of documents containing the term)
    ///
    /// # Flooring
    /// IDF is floored at 0.0 to prevent negative scores for high-frequency terms
    /// (terms appearing in > 50% of documents).
    fn compute_idf(&self, total_docs: u32, doc_freq: u32) -> f64 {
        let n = doc_freq as f64;
        let n = n.max(0.5); // Prevent division by zero
        let n_docs = total_docs as f64;

        // Standard BM25 IDF formula (without the +1 to allow negative values that we floor)
        let idf = ((n_docs - n + 0.5) / (n + 0.5)).ln();

        // Floor at 0 per D-03 - terms in >50% of docs get 0 IDF
        idf.max(0.0)
    }

    /// Compute TF saturation component
    ///
    /// Formula: (f × (k1 + 1)) / (f + k1 × (1 - b + b × (dl / avgdl)))
    ///
    /// Where:
    /// - f = term frequency in document
    /// - dl = document length (number of terms)
    /// - avgdl = average document length
    /// - k1 = term frequency saturation parameter
    /// - b = length normalization parameter
    ///
    /// # Characteristics
    /// - As f increases, the score increases but saturates (non-linear growth)
    /// - Longer documents (dl > avgdl) get lower scores when b > 0
    /// - When b = 0, document length has no effect
    fn compute_tf_component(&self, term_freq: u32, doc_length: u32, avgdl: f64) -> f64 {
        let f = term_freq as f64;
        let dl = doc_length as f64;
        let k1 = self.config.k1;
        let b = self.config.b;

        // Avoid division by zero for empty documents
        let avgdl = avgdl.max(1.0);

        let denominator = f + k1 * (1.0 - b + b * (dl / avgdl));

        (f * (k1 + 1.0)) / denominator
    }

    /// Compute BM25 score for a single term-document pair
    ///
    /// # Arguments
    /// - `term_freq`: Term frequency in the document (f)
    /// - `doc_length`: Document length in terms (dl)
    /// - `avgdl`: Average document length in the segment
    /// - `doc_freq`: Document frequency - number of docs containing the term (n)
    /// - `total_docs`: Total number of documents in the segment (N)
    ///
    /// # Returns
    /// BM25 score as f32 for API consistency with vector similarity scores (D-06)
    ///
    /// # Formula
    /// Score = IDF × TF_component
    ///
    /// # Note
    /// All intermediate calculations use f64 for precision (D-05).
    /// The final result is cast to f32 at the API boundary.
    pub fn score(
        &self,
        term_freq: u32,
        doc_length: u32,
        avgdl: f64,
        doc_freq: u32,
        total_docs: u32,
    ) -> f32 {
        let idf = self.compute_idf(total_docs, doc_freq);
        let tf_component = self.compute_tf_component(term_freq, doc_length, avgdl);

        let score = idf * tf_component;

        debug!(
            "BM25 score: tf={}, dl={}, avgdl={}, df={}, N={}, idf={}, tf_comp={}, score={}",
            term_freq, doc_length, avgdl, doc_freq, total_docs, idf, tf_component, score
        );

        score as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_scorer() -> BM25Scorer {
        let config = BM25Config::default();
        BM25Scorer::new(config)
    }

    #[test]
    fn test_idf_flooring() {
        let scorer = BM25Scorer::new(BM25Config::default());

        // When term appears in >50% of docs, IDF should floor at 0
        let idf = scorer.compute_idf(100, 60); // 60/100 = 60% > 50%
        assert_eq!(
            idf, 0.0,
            "IDF should be floored at 0 for high-frequency terms (>50%)"
        );

        // When term appears in exactly 50% of docs, IDF = ln(1) = 0
        let idf = scorer.compute_idf(100, 50);
        assert_eq!(idf, 0.0, "IDF should be 0 at exactly 50% (ln(1) = 0)");

        // When term appears in <50% of docs
        let idf = scorer.compute_idf(100, 10);
        assert!(idf > 0.0, "IDF should be positive for rare terms (<50%)");

        // Edge case: term in all docs
        let idf = scorer.compute_idf(100, 100);
        assert_eq!(idf, 0.0, "IDF should be 0 when term appears in all docs");
    }

    #[test]
    fn test_idf_flooring_near_100_percent() {
        let scorer = create_scorer();

        // When term appears in >50% of docs, IDF should be 0 (floored)
        let idf_60 = scorer.compute_idf(100, 60); // 60/100 = 60%
        let idf_90 = scorer.compute_idf(100, 90); // 90/100 = 90%
        let idf_99 = scorer.compute_idf(100, 99); // 99/100 = 99%
        let idf_100 = scorer.compute_idf(100, 100); // 100/100 = 100%

        // All should be floored to 0 since >50%
        assert_eq!(idf_60, 0.0, "IDF for 60% should be 0");
        assert_eq!(idf_90, 0.0, "IDF for 90% should be 0");
        assert_eq!(idf_99, 0.0, "IDF for 99% should be 0");
        assert_eq!(idf_100, 0.0, "IDF for 100% should be 0");
    }

    #[test]
    fn test_idf_at_50_percent_is_zero() {
        let scorer = create_scorer();

        // When term appears in exactly 50% of docs, IDF = ln(1.0) = 0
        let idf = scorer.compute_idf(100, 50);
        assert_eq!(idf, 0.0, "IDF at exactly 50% should be ln(1) = 0");
    }

    #[test]
    fn test_idf_positive_for_rare_terms() {
        let scorer = create_scorer();

        // When term appears in < 50% of docs, IDF should be positive
        let idf = scorer.compute_idf(100, 10); // 10/100 = 10%
        assert!(idf > 0.0, "IDF should be positive for rare terms");

        // Test at 49% (just below 50%)
        let idf_49 = scorer.compute_idf(100, 49); // 49/100 = 49%
        assert!(idf_49 > 0.0, "IDF should be positive just below 50%");
    }

    #[test]
    fn test_idf_formula_correctness() {
        let scorer = BM25Scorer::new(BM25Config::default());

        // Test case: N=1000, n=1 (rare term)
        // IDF = ln((1000-1+0.5)/(1+0.5)) = ln(666.333...) ≈ 6.502
        let idf = scorer.compute_idf(1000, 1);
        let inner: f64 = (1000.0 - 1.0 + 0.5) / (1.0 + 0.5);
        let expected = inner.ln();
        assert!(
            (idf - expected).abs() < 0.0001,
            "IDF formula incorrect: got {}, expected {}",
            idf,
            expected
        );

        // Test case: N=100, n=50 (50% of docs)
        // IDF = ln((100-50+0.5)/(50+0.5)) = ln(1.0) = 0.0
        let idf = scorer.compute_idf(100, 50);
        let inner: f64 = (100.0 - 50.0 + 0.5) / (50.0 + 0.5);
        let expected = inner.ln();
        assert!(
            (idf - expected).abs() < 0.0001,
            "IDF formula incorrect at 50%"
        );
    }

    #[test]
    fn test_tf_saturation() {
        // Term frequency should saturate, not grow linearly
        let config = BM25Config::new(1.2, 0.75).unwrap();
        let scorer = BM25Scorer::new(config);

        let tf1 = scorer.compute_tf_component(1, 100, 100.0);
        let tf10 = scorer.compute_tf_component(10, 100, 100.0);
        let tf100 = scorer.compute_tf_component(100, 100, 100.0);

        // 10x TF increase should result in much less than 10x score increase
        let ratio_1_to_10 = tf10 / tf1;
        let ratio_1_to_100 = tf100 / tf1;

        assert!(
            ratio_1_to_10 < 10.0,
            "TF should saturate: ratio was {}",
            ratio_1_to_10
        );
        assert!(
            ratio_1_to_100 < 100.0,
            "TF should saturate: ratio was {}",
            ratio_1_to_100
        );
        assert!(ratio_1_to_10 > 1.0, "But should still increase");
        assert!(
            ratio_1_to_100 > ratio_1_to_10,
            "More TF should give higher score"
        );
    }

    #[test]
    fn test_tf_saturation_k1_effect() {
        // Higher k1 = slower saturation
        let low_k1 = BM25Config::new(0.5, 0.75).unwrap();
        let high_k1 = BM25Config::new(2.0, 0.75).unwrap();

        let scorer_low = BM25Scorer::new(low_k1);
        let scorer_high = BM25Scorer::new(high_k1);

        let ratio_low_k1 = scorer_low.compute_tf_component(10, 100, 100.0)
            / scorer_low.compute_tf_component(1, 100, 100.0);
        let ratio_high_k1 = scorer_high.compute_tf_component(10, 100, 100.0)
            / scorer_high.compute_tf_component(1, 100, 100.0);

        // Higher k1 should have higher ratio (saturates slower)
        assert!(
            ratio_high_k1 > ratio_low_k1,
            "Higher k1 should saturate slower"
        );
    }

    #[test]
    fn test_tf_component_dl_equals_avgdl() {
        let config = BM25Config::new(1.2, 0.75).unwrap();
        let scorer = BM25Scorer::new(config);

        // When dl == avgdl, the formula simplifies to: (f * (k1 + 1)) / (f + k1)
        let tf = scorer.compute_tf_component(5, 100, 100.0);

        // Expected: (5 * 2.2) / (5 + 1.2) = 11 / 6.2 ≈ 1.774
        let expected = (5.0 * 2.2) / (5.0 + 1.2);
        assert!((tf - expected).abs() < 0.001);
    }

    #[test]
    fn test_tf_component_shorter_doc_higher_score() {
        let config = BM25Config::new(1.2, 0.75).unwrap();
        let scorer = BM25Scorer::new(config);

        // Same term frequency, different document lengths
        let tf_short = scorer.compute_tf_component(5, 50, 100.0);
        let tf_long = scorer.compute_tf_component(5, 200, 100.0);

        // Shorter document should get higher score
        assert!(tf_short > tf_long);
    }

    #[test]
    fn test_length_normalization() {
        // Longer documents should get lower scores (with b > 0)
        let config = BM25Config::new(1.2, 0.75).unwrap();
        let scorer = BM25Scorer::new(config);

        // Same term frequency (5), different document lengths, avgdl=100
        let score_short = scorer.score(5, 50, 100.0, 10, 1000); // dl=50 < avgdl
        let score_avg = scorer.score(5, 100, 100.0, 10, 1000); // dl=100 = avgdl
        let score_long = scorer.score(5, 200, 100.0, 10, 1000); // dl=200 > avgdl

        assert!(
            score_short > score_avg,
            "Shorter doc should have higher score"
        );
        assert!(score_avg > score_long, "Longer doc should have lower score");
    }

    #[test]
    fn test_length_normalization_with_default_b() {
        let config = BM25Config::new(1.2, 0.75).unwrap();
        let scorer = BM25Scorer::new(config);

        // Same term frequency, different document lengths
        let score_short = scorer.score(5, 50, 100.0, 10, 1000);
        let score_long = scorer.score(5, 200, 100.0, 10, 1000);

        assert!(
            score_short > score_long,
            "Longer doc should have lower score with b > 0"
        );
    }

    #[test]
    fn test_no_normalization_when_b_is_zero() {
        let config = BM25Config::new(1.2, 0.0).unwrap();
        let scorer = BM25Scorer::new(config);

        let score_short = scorer.score(5, 50, 100.0, 10, 1000);
        let score_long = scorer.score(5, 200, 100.0, 10, 1000);

        // With b=0, scores should be approximately equal (no length normalization)
        let diff = (score_short - score_long).abs();
        assert!(
            diff < 0.001,
            "With b=0, scores should be equal: short={}, long={}, diff={}",
            score_short,
            score_long,
            diff
        );
    }

    #[test]
    fn test_full_normalization_when_b_is_one() {
        // When b=1, full length normalization
        let config = BM25Config::new(1.2, 1.0).unwrap();
        let scorer = BM25Scorer::new(config);

        // With b=1: TF_component = (f * (k1+1)) / (f + k1 * (dl / avgdl))
        // At dl=avgdl, this equals (f * (k1+1)) / (f + k1)
        let score_at_avgdl = scorer.compute_tf_component(5, 100, 100.0);
        let expected_at_avgdl = (5.0 * 2.2) / (5.0 + 1.2); // f=5, k1=1.2
        assert!(
            (score_at_avgdl - expected_at_avgdl).abs() < 0.0001,
            "TF at avgdl incorrect"
        );

        // Test scores at different lengths with b=1
        let score_short = scorer.score(5, 50, 100.0, 10, 1000);
        let score_avg = scorer.score(5, 100, 100.0, 10, 1000);
        let score_long = scorer.score(5, 200, 100.0, 10, 1000);

        // Full normalization should show clear distinction
        assert!(
            score_short > score_avg,
            "Short doc should score higher with b=1"
        );
        assert!(
            score_avg > score_long,
            "Long doc should score lower with b=1"
        );
    }

    #[test]
    fn test_bm25_score_positive() {
        let config = BM25Config::default();
        let scorer = BM25Scorer::new(config);

        // Scenario: term appears 3 times, dl=100, avgdl=80, df=10, N=1000
        let score = scorer.score(3, 100, 80.0, 10, 1000);

        // Score should be positive
        assert!(score > 0.0, "BM25 score should be positive");

        // Verify it's a reasonable f32 value
        assert!(score.is_finite(), "Score should be finite");
        assert!(!score.is_nan(), "Score should not be NaN");
    }

    #[test]
    fn test_bm25_score_comparison() {
        let config = BM25Config::default();
        let scorer = BM25Scorer::new(config);

        // Verify score increases with term frequency
        let score_tf1 = scorer.score(1, 100, 80.0, 10, 1000);
        let score_tf3 = scorer.score(3, 100, 80.0, 10, 1000);
        let score_tf10 = scorer.score(10, 100, 80.0, 10, 1000);

        assert!(score_tf3 > score_tf1, "Higher TF should give higher score");
        assert!(score_tf10 > score_tf3, "Higher TF should give higher score");

        // Verify rare terms give higher scores than common terms
        let score_rare = scorer.score(5, 100, 100.0, 5, 1000); // df=5, rare
        let score_common = scorer.score(5, 100, 100.0, 500, 1000); // df=500, common

        assert!(
            score_rare > score_common,
            "Rare term should have higher IDF and score"
        );
    }

    #[test]
    fn test_bm25_score_rare_vs_common_terms() {
        let config = BM25Config::default();
        let scorer = BM25Scorer::new(config);

        // Same TF and DL, different document frequencies (all <50% to have non-zero IDF)
        let score_very_rare = scorer.score(5, 100, 100.0, 1, 1000); // 0.1% - very rare
        let score_rare = scorer.score(5, 100, 100.0, 10, 1000); // 1% - rare
        let score_uncommon = scorer.score(5, 100, 100.0, 100, 1000); // 10% - uncommon
        let score_moderate = scorer.score(5, 100, 100.0, 400, 1000); // 40% - moderate (near threshold)

        assert!(
            score_very_rare > score_rare,
            "Very rare should score higher than rare"
        );
        assert!(
            score_rare > score_uncommon,
            "Rare should score higher than uncommon"
        );
        assert!(
            score_uncommon > score_moderate,
            "Uncommon should score higher than moderate"
        );
    }

    #[test]
    fn test_score_returns_f32() {
        let scorer = create_scorer();

        let score = scorer.score(3, 100, 80.0, 10, 1000);

        // Verify it's f32
        let _f32_check: f32 = score;
        assert!(score.is_finite());
    }

    #[test]
    fn test_score_positive_for_normal_case() {
        let scorer = create_scorer();

        // Typical scenario
        let score = scorer.score(3, 100, 80.0, 10, 1000);

        assert!(score > 0.0);
    }

    #[test]
    fn test_score_decreases_with_high_df() {
        let scorer = create_scorer();

        // Score should decrease as document frequency increases (less rare = lower IDF)
        // All must be <50% of total docs to have non-zero IDF
        let score_df10 = scorer.score(5, 100, 100.0, 10, 100); // 10% - low DF
        let score_df30 = scorer.score(5, 100, 100.0, 30, 100); // 30% - medium DF
        let score_df49 = scorer.score(5, 100, 100.0, 49, 100); // 49% - high DF but still <50%

        assert!(score_df30 < score_df10, "Higher DF should give lower score");
        assert!(
            score_df49 < score_df30,
            "Even higher DF should give even lower score"
        );

        // At 50% or above, IDF should be 0 (floored)
        let score_df50 = scorer.score(5, 100, 100.0, 50, 100); // 50% - IDF = 0
        let score_df60 = scorer.score(5, 100, 100.0, 60, 100); // 60% - IDF = 0 (floored)
        assert_eq!(
            score_df50, 0.0,
            "Score at 50% should be 0 (IDF = ln(1) = 0)"
        );
        assert_eq!(
            score_df60, 0.0,
            "Score at 60% should be 0 (IDF floored to 0)"
        );
    }

    #[test]
    fn test_score_increases_with_term_freq() {
        let scorer = create_scorer();

        let score_tf1 = scorer.score(1, 100, 100.0, 10, 1000);
        let score_tf3 = scorer.score(3, 100, 100.0, 10, 1000);
        let score_tf10 = scorer.score(10, 100, 100.0, 10, 1000);

        assert!(score_tf3 > score_tf1, "Higher TF should give higher score");
        assert!(score_tf10 > score_tf3, "Higher TF should give higher score");
    }

    #[test]
    fn test_score_with_custom_k1() {
        // Higher k1 = slower saturation
        let config_low_k1 = BM25Config::new(0.5, 0.75).unwrap();
        let config_high_k1 = BM25Config::new(2.0, 0.75).unwrap();

        let scorer_low_k1 = BM25Scorer::new(config_low_k1);
        let scorer_high_k1 = BM25Scorer::new(config_high_k1);

        // With high k1, TF saturates more slowly, so high TF should give relatively higher score
        let score_low_k1 = scorer_low_k1.score(10, 100, 100.0, 10, 1000);
        let score_high_k1 = scorer_high_k1.score(10, 100, 100.0, 10, 1000);

        // Higher k1 should give higher score for high TF
        assert!(score_high_k1 > score_low_k1);
    }

    #[test]
    fn test_empty_document() {
        let scorer = create_scorer();

        // Empty document (dl=0, tf=0)
        let score = scorer.score(0, 0, 100.0, 10, 1000);

        // Should not panic and should return a valid (possibly zero) score
        assert!(score.is_finite());
    }

    #[test]
    fn test_term_not_in_document() {
        let scorer = create_scorer();

        // Document with length but term frequency is 0
        let score = scorer.score(0, 100, 100.0, 10, 1000);

        // TF component should be 0, so score should be 0
        assert_eq!(score, 0.0);
    }
}
