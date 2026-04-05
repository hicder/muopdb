/// Score Normalization for BM25 and Hybrid Search
///
/// Provides utilities for normalizing scores to a common [0,1] range
/// to enable proper fusion of BM25 text relevance scores with vector
/// similarity scores.
/// Statistics for a set of scores
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoreStats {
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub std: f32,
    pub count: usize,
}

impl ScoreStats {
    /// Create empty statistics for empty input
    pub fn empty() -> Self {
        Self {
            min: 0.0,
            max: 0.0,
            mean: 0.0,
            std: 0.0,
            count: 0,
        }
    }

    /// Check if stats are valid (non-empty input)
    pub fn is_valid(&self) -> bool {
        self.count > 0
    }
}

/// Min-max score normalizer
///
/// Maps scores from any range to [0, 1] using the formula:
/// normalized = (score - min) / (max - min)
///
/// Handles edge cases:
/// - Empty input: returns empty vector
/// - Single value: returns [1.0]
/// - All identical values: returns all [0.0]
/// - NaN values: filtered out before processing
#[derive(Debug, Clone)]
pub struct ScoreNormalizer;

impl ScoreNormalizer {
    /// Create a new ScoreNormalizer
    pub fn new() -> Self {
        Self
    }

    /// Normalize scores to [0, 1] range using min-max normalization
    ///
    /// Formula: (score - min) / (max - min)
    ///
    /// # Arguments
    /// * `scores` - Slice of f32 scores to normalize
    ///
    /// # Returns
    /// Vector of normalized scores in [0, 1] range
    ///
    /// # Edge Cases
    /// - Empty input: returns empty vector
    /// - Single value: returns vec![1.0]
    /// - All identical values: returns all zeros
    /// - NaN values: filtered out before normalization
    ///
    /// # Examples
    /// ```
    /// use index::bm25::ScoreNormalizer;
    ///
    /// let normalizer = ScoreNormalizer::new();
    /// let normalized = normalizer.normalize_scores(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    /// assert_eq!(normalized, vec![0.0, 0.25, 0.5, 0.75, 1.0]);
    /// ```
    pub fn normalize_scores(&self, scores: &[f32]) -> Vec<f32> {
        // Filter out NaN values
        let valid_scores: Vec<f32> = scores.iter().filter(|&&s| !s.is_nan()).copied().collect();

        if valid_scores.is_empty() {
            return Vec::new();
        }

        if valid_scores.len() == 1 {
            return vec![1.0];
        }

        let min = valid_scores.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max = valid_scores
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // If all scores are the same, return all zeros
        if (max - min).abs() < f32::EPSILON {
            return valid_scores.iter().map(|_| 0.0).collect();
        }

        valid_scores
            .iter()
            .map(|&score| (score - min) / (max - min))
            .collect()
    }

    /// Normalize multiple score lists independently
    ///
    /// Useful for normalizing scores across multiple segments or users
    /// where each score list should be normalized independently.
    ///
    /// # Arguments
    /// * `score_lists` - Slice of score vectors to normalize
    ///
    /// # Returns
    /// Vector of normalized score vectors
    pub fn batch_normalize(&self, score_lists: &[Vec<f32>]) -> Vec<Vec<f32>> {
        score_lists
            .iter()
            .map(|scores| self.normalize_scores(scores))
            .collect()
    }

    /// Compute statistics for a set of scores
    ///
    /// Useful for debugging and diagnostics to understand score distribution.
    ///
    /// # Arguments
    /// * `scores` - Slice of f32 scores
    ///
    /// # Returns
    /// ScoreStats containing min, max, mean, std, and count
    pub fn score_stats(&self, scores: &[f32]) -> ScoreStats {
        let valid_scores: Vec<f32> = scores.iter().filter(|&&s| !s.is_nan()).copied().collect();

        if valid_scores.is_empty() {
            return ScoreStats::empty();
        }

        let count = valid_scores.len();
        let min = valid_scores.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max = valid_scores
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        let sum: f32 = valid_scores.iter().sum();
        let mean = sum / count as f32;

        let variance: f32 = valid_scores
            .iter()
            .map(|&s| (s - mean).powi(2))
            .sum::<f32>()
            / count as f32;
        let std = variance.sqrt();

        ScoreStats {
            min,
            max,
            mean,
            std,
            count,
        }
    }

    /// Normalize scores with outlier clipping
    ///
    /// First clips outliers beyond the specified number of standard deviations,
    /// then applies min-max normalization. Useful when BM25 scores have extreme
    /// spikes that would compress the normalized range.
    ///
    /// # Arguments
    /// * `scores` - Slice of f32 scores to normalize
    /// * `std_devs` - Number of standard deviations for clipping threshold
    ///
    /// # Returns
    /// Vector of normalized scores with outliers clipped
    pub fn normalize_with_clipping(&self, scores: &[f32], std_devs: f32) -> Vec<f32> {
        let valid_scores: Vec<f32> = scores.iter().filter(|&&s| !s.is_nan()).copied().collect();

        if valid_scores.is_empty() {
            return Vec::new();
        }

        if valid_scores.len() == 1 {
            return vec![1.0];
        }

        let stats = self.score_stats(&valid_scores);

        // Clip outliers
        let lower_bound = stats.mean - std_devs * stats.std;
        let upper_bound = stats.mean + std_devs * stats.std;

        let clipped_scores: Vec<f32> = valid_scores
            .iter()
            .map(|&s| s.clamp(lower_bound, upper_bound))
            .collect();

        self.normalize_scores(&clipped_scores)
    }
}

impl Default for ScoreNormalizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function for min-max normalization
///
/// Maps scores from any range to [0, 1] using min-max normalization.
/// This is a standalone function equivalent to `ScoreNormalizer::new().normalize_scores(scores)`.
///
/// # Arguments
/// * `scores` - Slice of f32 scores to normalize
///
/// # Returns
/// Vector of normalized scores in [0, 1] range
///
/// # Examples
/// ```
/// use index::bm25::normalize_scores;
///
/// let normalized = normalize_scores(&[1.0, 2.0, 3.0, 4.0, 5.0]);
/// assert_eq!(normalized, vec![0.0, 0.25, 0.5, 0.75, 1.0]);
/// ```
pub fn normalize_scores(scores: &[f32]) -> Vec<f32> {
    ScoreNormalizer::new().normalize_scores(scores)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test 1: Basic normalization [1.0, 2.0, 3.0, 4.0, 5.0] -> [0.0, 0.25, 0.5, 0.75, 1.0]
    #[test]
    fn test_normalize_scores_basic() {
        let normalizer = ScoreNormalizer::new();
        let scores = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let normalized = normalizer.normalize_scores(&scores);
        assert_eq!(normalized, vec![0.0, 0.25, 0.5, 0.75, 1.0]);
    }

    // Test 2: Single score becomes 1.0
    #[test]
    fn test_normalize_scores_single_value() {
        let normalizer = ScoreNormalizer::new();
        let scores = vec![5.0];
        let normalized = normalizer.normalize_scores(&scores);
        assert_eq!(normalized, vec![1.0]);
    }

    // Test 3: All same values map to 0.0
    #[test]
    fn test_normalize_scores_all_identical() {
        let normalizer = ScoreNormalizer::new();
        let scores = vec![2.0, 2.0, 2.0];
        let normalized = normalizer.normalize_scores(&scores);
        assert_eq!(normalized, vec![0.0, 0.0, 0.0]);
    }

    // Test 4: Empty input returns empty
    #[test]
    fn test_normalize_scores_empty() {
        let normalizer = ScoreNormalizer::new();
        let scores: Vec<f32> = vec![];
        let normalized = normalizer.normalize_scores(&scores);
        assert!(normalized.is_empty());
    }

    // Test 5: Negative values [-5.0, 0.0, 5.0] -> [0.0, 0.5, 1.0]
    #[test]
    fn test_normalize_scores_negative_values() {
        let normalizer = ScoreNormalizer::new();
        let scores = vec![-5.0, 0.0, 5.0];
        let normalized = normalizer.normalize_scores(&scores);
        assert_eq!(normalized, vec![0.0, 0.5, 1.0]);
    }

    // Test 6: NaN values are filtered out
    #[test]
    fn test_normalize_scores_handles_nan() {
        let normalizer = ScoreNormalizer::new();
        let scores = vec![1.0, f32::NAN, 2.0, f32::NAN, 3.0];
        let normalized = normalizer.normalize_scores(&scores);
        assert_eq!(normalized, vec![0.0, 0.5, 1.0]);
    }

    // Test: Convenience function works
    #[test]
    fn test_normalize_scores_convenience_function() {
        let scores = vec![1.0, 2.0, 3.0];
        let normalized = normalize_scores(&scores);
        assert_eq!(normalized, vec![0.0, 0.5, 1.0]);
    }

    // Test: Batch normalization
    #[test]
    fn test_batch_normalize() {
        let normalizer = ScoreNormalizer::new();
        let score_lists = vec![vec![1.0, 2.0, 3.0], vec![10.0, 20.0], vec![5.0]];
        let normalized = normalizer.batch_normalize(&score_lists);
        assert_eq!(normalized.len(), 3);
        assert_eq!(normalized[0], vec![0.0, 0.5, 1.0]);
        assert_eq!(normalized[1], vec![0.0, 1.0]);
        assert_eq!(normalized[2], vec![1.0]);
    }

    // Test: Score statistics
    #[test]
    fn test_score_stats() {
        let normalizer = ScoreNormalizer::new();
        let scores = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = normalizer.score_stats(&scores);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
        assert_eq!(stats.count, 5);
        // Mean should be 3.0
        assert!((stats.mean - 3.0).abs() < 0.0001);
        // For uniform 1-5 distribution, std is sqrt(2) ≈ 1.414
        let expected_std = 2.0f32.sqrt();
        assert!((stats.std - expected_std).abs() < 0.0001);
    }

    // Test: Score stats with empty input
    #[test]
    fn test_score_stats_empty() {
        let normalizer = ScoreNormalizer::new();
        let scores: Vec<f32> = vec![];
        let stats = normalizer.score_stats(&scores);
        assert!(!stats.is_valid());
        assert_eq!(stats.count, 0);
    }

    // Test: Score stats with NaN
    #[test]
    fn test_score_stats_with_nan() {
        let normalizer = ScoreNormalizer::new();
        let scores = vec![1.0, f32::NAN, 3.0];
        let stats = normalizer.score_stats(&scores);
        assert_eq!(stats.count, 2);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 3.0);
    }

    // Test: Normalization with outlier clipping
    #[test]
    fn test_normalize_with_clipping() {
        let normalizer = ScoreNormalizer::new();
        // One extreme outlier at 100, rest in 1-10 range
        let scores = vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0];
        let normalized = normalizer.normalize_with_clipping(&scores, 2.0);

        // Without clipping, 1.0 would be ~0.0 and 100.0 would be 1.0
        // With clipping at 2 std devs, the extreme value should be clipped
        // This should give better spread for the normal values
        let stats = normalizer.score_stats(&scores);
        let _lower = stats.mean - 2.0 * stats.std;
        let upper = stats.mean + 2.0 * stats.std;

        // 100 should be clipped to upper bound
        assert!(upper < 100.0, "Outlier should be clipped");

        // All normalized values should be in [0, 1]
        for &val in &normalized {
            assert!(
                (0.0..=1.0).contains(&val),
                "Value {} should be in [0, 1]",
                val
            );
        }
    }

    // Test: Empty input for clipping
    #[test]
    fn test_normalize_with_clipping_empty() {
        let normalizer = ScoreNormalizer::new();
        let scores: Vec<f32> = vec![];
        let normalized = normalizer.normalize_with_clipping(&scores, 3.0);
        assert!(normalized.is_empty());
    }

    // Test: Single value for clipping
    #[test]
    fn test_normalize_with_clipping_single() {
        let normalizer = ScoreNormalizer::new();
        let scores = vec![42.0];
        let normalized = normalizer.normalize_with_clipping(&scores, 3.0);
        assert_eq!(normalized, vec![1.0]);
    }

    // Test: Verify normalized values are within [0, 1]
    #[test]
    fn test_normalized_range() {
        let normalizer = ScoreNormalizer::new();
        let scores = vec![-100.0, -50.0, 0.0, 50.0, 100.0];
        let normalized = normalizer.normalize_scores(&scores);

        for &val in &normalized {
            assert!(val >= 0.0, "Value {} should be >= 0", val);
            assert!(val <= 1.0, "Value {} should be <= 1", val);
        }

        assert_eq!(normalized[0], 0.0); // min should be 0
        assert_eq!(normalized[normalized.len() - 1], 1.0); // max should be 1
    }

    // Test: Default implementation
    #[test]
    fn test_default_normalizer() {
        let normalizer = ScoreNormalizer;
        let scores = vec![1.0, 2.0, 3.0];
        let normalized = normalizer.normalize_scores(&scores);
        assert_eq!(normalized, vec![0.0, 0.5, 1.0]);
    }
}
