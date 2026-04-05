use anyhow::{Error, Result};
use serde::{Deserialize, Serialize};

/// Default k1 parameter (term frequency saturation)
/// Lucene/Elasticsearch default: 1.2
pub const DEFAULT_K1: f64 = 1.2;

/// Default b parameter (length normalization)
/// Lucene/Elasticsearch default: 0.75
pub const DEFAULT_B: f64 = 0.75;

/// Configuration for BM25 scoring parameters.
///
/// BM25 (Best Matching 25) is a probabilistic ranking function used to rank
/// documents by relevance to a given search query.
///
/// # Parameters
/// - `k1`: Controls term frequency saturation (how quickly TF saturates)
/// - `b`: Controls length normalization (0 = no normalization, 1 = full normalization)
///
/// # Defaults
/// Uses Lucene/Elasticsearch defaults: k1 = 1.2, b = 0.75
///
/// # Example
/// ```
/// use index::bm25::BM25Config;
///
/// // Default configuration
/// let config = BM25Config::default();
/// assert_eq!(config.k1, 1.2);
/// assert_eq!(config.b, 0.75);
///
/// // Custom configuration with validation
/// let config = BM25Config::new(1.5, 0.5).unwrap();
/// ```
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct BM25Config {
    /// Term frequency saturation parameter
    /// Controls how quickly term frequency saturates
    /// Higher values = slower saturation (linear TF growth)
    /// Default: 1.2 (Lucene/Elasticsearch default)
    pub k1: f64,

    /// Length normalization parameter
    /// 0 = no normalization, 1 = full normalization
    /// Default: 0.75 (Lucene/Elasticsearch default)
    pub b: f64,
}

impl BM25Config {
    /// Validate configuration parameters
    ///
    /// # Validation Rules
    /// - k1 must be > 0
    /// - b must be in [0, 1]
    ///
    /// # Returns
    /// - `Ok(())` if parameters are valid
    /// - `Err` with descriptive message if parameters are invalid
    pub fn validate(&self) -> Result<()> {
        if self.k1 <= 0.0 {
            return Err(Error::msg("k1 must be > 0"));
        }
        if self.b < 0.0 || self.b > 1.0 {
            return Err(Error::msg("b must be in [0, 1]"));
        }
        Ok(())
    }

    /// Create a new BM25Config with custom parameters
    ///
    /// # Arguments
    /// - `k1`: Term frequency saturation parameter (must be > 0)
    /// - `b`: Length normalization parameter (must be in [0, 1])
    ///
    /// # Returns
    /// - `Ok(BM25Config)` if parameters are valid
    /// - `Err` if parameters fail validation
    pub fn new(k1: f64, b: f64) -> Result<Self> {
        let config = Self { k1, b };
        config.validate()?;
        Ok(config)
    }
}

impl Default for BM25Config {
    fn default() -> Self {
        Self {
            k1: DEFAULT_K1,
            b: DEFAULT_B,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = BM25Config::default();
        assert_eq!(config.k1, 1.2);
        assert_eq!(config.b, 0.75);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_valid_config() {
        let config = BM25Config::new(1.5, 0.5).unwrap();
        assert_eq!(config.k1, 1.5);
        assert_eq!(config.b, 0.5);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validation_k1_positive() {
        // k1 must be > 0 - test boundary cases per plan
        assert!(BM25Config::new(0.0, 0.5).is_err(), "k1=0 should fail");
        assert!(BM25Config::new(-1.0, 0.5).is_err(), "k1=-1.0 should fail");
        assert!(
            BM25Config::new(-0.0001, 0.5).is_err(),
            "k1=-0.0001 should fail"
        );
        assert!(BM25Config::new(1.0, 0.5).is_ok(), "k1=1.0 should succeed");
        assert!(
            BM25Config::new(0.0001, 0.5).is_ok(),
            "k1=0.0001 should succeed"
        );
    }

    #[test]
    fn test_config_validation_b_range() {
        // b must be in [0, 1] - test boundary cases per plan
        assert!(BM25Config::new(1.2, -0.1).is_err(), "b=-0.1 should fail");
        assert!(BM25Config::new(1.2, 1.1).is_err(), "b=1.1 should fail");
        assert!(BM25Config::new(1.2, 0.0).is_ok(), "b=0.0 should succeed");
        assert!(BM25Config::new(1.2, 1.0).is_ok(), "b=1.0 should succeed");
        assert!(BM25Config::new(1.2, 0.75).is_ok(), "b=0.75 should succeed");
        assert!(BM25Config::new(1.2, 0.5).is_ok(), "b=0.5 should succeed");
    }

    #[test]
    fn test_config_validation_error_messages() {
        let invalid_k1 = BM25Config::new(0.0, 0.5);
        assert!(invalid_k1.is_err());
        assert!(
            invalid_k1
                .unwrap_err()
                .to_string()
                .contains("k1 must be > 0"),
            "Error message should mention k1 > 0"
        );

        let invalid_b_low = BM25Config::new(1.2, -0.1);
        assert!(invalid_b_low.is_err());
        assert!(
            invalid_b_low
                .unwrap_err()
                .to_string()
                .contains("b must be in [0, 1]"),
            "Error message should mention b in [0, 1]"
        );

        let invalid_b_high = BM25Config::new(1.2, 1.1);
        assert!(invalid_b_high.is_err());
        assert!(
            invalid_b_high
                .unwrap_err()
                .to_string()
                .contains("b must be in [0, 1]"),
            "Error message should mention b in [0, 1]"
        );
    }

    #[test]
    fn test_config_equality() {
        let config1 = BM25Config::new(1.2, 0.75).unwrap();
        let config2 = BM25Config::new(1.2, 0.75).unwrap();
        assert_eq!(config1, config2);

        let config3 = BM25Config::new(1.5, 0.5).unwrap();
        assert_ne!(config1, config3);
    }

    #[test]
    fn test_config_serialization_yaml() {
        // Test YAML serialization as specified in the plan
        let config = BM25Config::new(1.2, 0.75).unwrap();
        let yaml = serde_yaml::to_string(&config).unwrap();
        assert!(yaml.contains("k1: 1.2"), "YAML should contain k1: 1.2");
        assert!(yaml.contains("b: 0.75"), "YAML should contain b: 0.75");

        let deserialized: BM25Config = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(config, deserialized);
    }

    #[test]
    fn test_config_serialization_json() {
        // Also test JSON for completeness
        let config = BM25Config::new(1.5, 0.6).unwrap();
        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: BM25Config = serde_json::from_str(&serialized).unwrap();
        assert_eq!(config, deserialized);
    }
}
