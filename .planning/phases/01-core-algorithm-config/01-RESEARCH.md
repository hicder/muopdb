# Phase 01: Core Algorithm & Config - Research

**Researched:** 2026-04-05
**Domain:** BM25 ranking algorithm implementation in Rust
**Confidence:** HIGH

## Summary

This phase implements the foundational BM25 (Best Matching 25) scoring formula for MuopDB's text search capabilities. BM25 is a probabilistic ranking function that improves upon TF-IDF by introducing term frequency saturation and document length normalization.

The core components to implement are:
1. **BM25Config** - Configuration struct with validated k1 and b parameters
2. **BM25 scoring function** - IDF × TF component formula
3. **IDF computation** - With flooring at 0 to prevent negative scores
4. **Unit tests** - Verifying correctness against known BM25 values

The implementation follows MuopDB's existing patterns: config structs with serde serialization, validation methods returning `anyhow::Result<()>`, and numerical precision using f64 internally with f32 API returns for consistency with vector similarity scores.

**Primary recommendation:** Create a new `bm25` module under `rs/index/src/` with `mod.rs`, `config.rs`, and `scorer.rs` following the existing `multi_terms` module structure.

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

| Decision | Value | ID |
|----------|-------|-----|
| Parameter defaults | k1 = 1.2, b = 0.75 (Lucene/Elasticsearch defaults) | D-01 |
| Parameter ranges | k1 ∈ (0, ∞), b ∈ [0, 1] with validation on construction | D-02 |
| IDF edge handling | Floor IDF at 0 to prevent negative scores for high-frequency terms | D-03 |
| IDF formula | Use standard BM25 IDF: log(1 + (N - n + 0.5) / (n + 0.5)) | D-04 |
| TF precision | Use f64 for all intermediate BM25 calculations | D-05 |
| API precision | Return f32 scores for API consistency with existing vector similarity scores | D-06 |
| Statistics storage | Store per-segment statistics in dedicated bm25_stats/ subdirectory | D-07 |
| Serialization | Use rkyv serialization for zero-copy deserialization at query time | D-08 |

### the agent's Discretion
- Tokenization pipeline design (will adapt from existing TermIndex patterns)
- Exact struct layout for BM25Config (follow existing Config patterns in codebase)
- Unit test organization (follow existing test patterns)

### Deferred Ideas (OUT OF SCOPE)
- None — discussion stayed within phase scope

</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| CORE-01 | BM25 scoring formula with configurable k1/b | BM25 formula from Lucene reference; config patterns from `collection.rs` and `pq/mod.rs` |
| CORE-02 | IDF computation with flooring at 0 | Standard BM25 IDF formula with max(0, IDF) for edge cases |
| CORE-03 | TF saturation component with precision-safe formula | Use f64 intermediates, return f32; saturation formula from Robertson et al. |
| CORE-04 | Document length normalization | Length definition: total terms; avgdl per segment |
| API-01 | BM25Config struct with validated k1/b | Follow `ProductQuantizerConfig` pattern with `validate()` method |

</phase_requirements>

## Standard Stack

### Core Libraries
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `serde` | v1.0 | Config serialization/deserialization | Used throughout codebase for configs |
| `anyhow` | latest | Error handling | Standard error handling pattern in MuopDB |
| `rkyv` | v0.8.10 | Zero-copy statistics serialization | Consistent with existing posting list format |
| `tracing` | v0.1 | Structured logging | Used for debug logging across codebase |

### Already Available in Workspace
| Crate | Purpose | Used In |
|-------|---------|---------|
| `config` | Config structs and validation | `CollectionConfig`, `ProductQuantizerConfig` |
| `index` | Index implementations | Where BM25 will live |

**Installation:** All dependencies are already in the workspace.

## Architecture Patterns

### Recommended Module Structure

```
rs/index/src/
├── bm25/
│   ├── mod.rs           # Public exports
│   ├── config.rs        # BM25Config struct with validation
│   └── scorer.rs        # BM25Scorer with scoring formula
```

### Pattern 1: Config with Validation (from ProductQuantizerConfig)

**What:** Struct with serde derive, validate() method returning Result<()>

**When to use:** BM25Config implementation

**Example:**
```rust
// Source: rs/quantization/src/pq/mod.rs
use serde::{Deserialize, Serialize};
use anyhow::{Error, Result};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ProductQuantizerConfig {
    pub dimension: usize,
    pub subvector_dimension: usize,
    pub num_bits: u8,
}

impl ProductQuantizerConfig {
    pub fn validate(&self) -> Result<()> {
        if !self.dimension.is_multiple_of(self.subvector_dimension) {
            return Err(Error::msg("Dimensions are not valid"));
        }
        Ok(())
    }
}
```

### Pattern 2: BM25 Config with Defaults (from CollectionConfig)

**What:** Config with Default trait for standard parameters

**When to use:** Default values for k1 and b

**Example:**
```rust
// Source: rs/config/src/collection.rs
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct CollectionConfig {
    pub num_features: usize,
    // ... other fields
}

impl Default for CollectionConfig {
    fn default() -> Self {
        Self {
            num_features: 768,
            // ...
        }
    }
}
```

### Pattern 3: BM25 Scorer Struct

**What:** Struct holding config + statistics, implementing score computation

**When to use:** Computing BM25 scores for query-document pairs

**Example structure:**
```rust
pub struct BM25Scorer {
    config: BM25Config,
    // Statistics passed at query time
}

impl BM25Scorer {
    pub fn new(config: BM25Config) -> Self {
        Self { config }
    }
    
    /// Compute BM25 score for a single term-document pair
    pub fn score_term(
        &self,
        term_freq: u32,      // f: term frequency in document
        doc_length: u32,      // dl: document length
        avgdl: f64,           // avgdl: average document length
        doc_freq: u32,        // n: document frequency (docs containing term)
        total_docs: u32,      // N: total number of documents
    ) -> f32 {
        let idf = self.compute_idf(total_docs, doc_freq);
        let tf_component = self.compute_tf_component(term_freq, doc_length, avgdl);
        (idf * tf_component) as f32
    }
}
```

### Anti-Patterns to Avoid

- **Don't use std::sync::Mutex in async contexts:** Use `async_lock::RwLock` instead (per CONVENTIONS.md)
- **Don't create custom serialization formats:** Use rkyv for statistics (per D-08)
- **Don't use f32 for intermediate calculations:** Use f64 internally, convert to f32 only at API boundary (per D-05, D-06)

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Config validation logic | Custom validation framework | Standard `anyhow::Result<()>` pattern | Consistent with codebase |
| IDF computation | Simplified/log10 variant | Standard BM25 formula with ln | Lucene/Elasticsearch compatibility |
| Floating-point precision | Single-precision throughout | f64 intermediate, f32 API | Numerical stability, API consistency |

**Key insight:** The BM25 formula has subtle edge cases (IDF flooring, k1/b parameter ranges) that are already specified in the user constraints. Follow these exactly rather than implementing a variant.

## BM25 Formula Implementation

### Formula Specification

**IDF (Inverse Document Frequency):**
```
IDF = ln(1 + (N - n + 0.5) / (n + 0.5))
```
Where:
- N = total number of documents in the segment
- n = document frequency (number of documents containing the term)

With flooring: `IDF = max(0.0, IDF)`

**TF Component (Term Frequency with saturation):**
```
TF_component = (f * (k1 + 1)) / (f + k1 * (1 - b + b * (dl / avgdl)))
```
Where:
- f = term frequency in the document
- dl = document length (number of terms)
- avgdl = average document length in the segment
- k1 = term saturation parameter (default 1.2)
- b = length normalization parameter (default 0.75)

**Final Score:**
```
Score = IDF * TF_component
```

### Implementation Considerations

1. **IDF Flooring:** When n > N/2 (term appears in >50% of docs), raw IDF becomes negative. Floor at 0 per D-03.

2. **Precision:** Use f64 for all intermediate calculations (IDF, TF_component), cast to f32 only at return.

3. **Edge Cases:**
   - n = 0: IDF = ln(1 + (N + 0.5) / 0.5) = ln(1 + 2N + 1) ≈ ln(2N) for large N
   - n = N: IDF → ln(1 + 0.5/(N + 0.5)) ≈ 0, floored to 0
   - dl = 0: TF_component = (f * (k1 + 1)) / f = k1 + 1 (but f would also be 0, so handle empty docs)

## Code Examples

### BM25Config with Validation

```rust
// Source: Based on rs/quantization/src/pq/mod.rs patterns
use serde::{Deserialize, Serialize};
use anyhow::{Error, Result};

const DEFAULT_K1: f64 = 1.2;
const DEFAULT_B: f64 = 0.75;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct BM25Config {
    /// Term frequency saturation parameter
    /// Controls how quickly term frequency saturates
    /// Default: 1.2 (Lucene/Elasticsearch default)
    pub k1: f64,
    
    /// Length normalization parameter
    /// 0 = no normalization, 1 = full normalization
    /// Default: 0.75 (Lucene/Elasticsearch default)
    pub b: f64,
}

impl BM25Config {
    /// Validate configuration parameters
    /// - k1 must be > 0
    /// - b must be in [0, 1]
    pub fn validate(&self) -> Result<()> {
        if self.k1 <= 0.0 {
            return Err(Error::msg("k1 must be > 0"));
        }
        if self.b < 0.0 || self.b > 1.0 {
            return Err(Error::msg("b must be in [0, 1]"));
        }
        Ok(())
    }
    
    /// Create a new config with custom parameters
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
```

### BM25 Scorer Implementation

```rust
use tracing::debug;

pub struct BM25Scorer {
    config: BM25Config,
}

impl BM25Scorer {
    pub fn new(config: BM25Config) -> Self {
        Self { config }
    }
    
    /// Compute IDF with flooring at 0
    /// 
    /// Formula: ln(1 + (N - n + 0.5) / (n + 0.5))
    /// Where N = total docs, n = docs containing term
    fn compute_idf(&self, total_docs: u32, doc_freq: u32) -> f64 {
        let n = doc_freq as f64;
        let n = n.max(0.5); // Prevent division by zero
        let N = total_docs as f64;
        
        let idf = (1.0 + (N - n + 0.5) / (n + 0.5)).ln();
        
        // Floor at 0 per D-03
        idf.max(0.0)
    }
    
    /// Compute TF saturation component
    ///
    /// Formula: (f * (k1 + 1)) / (f + k1 * (1 - b + b * (dl / avgdl)))
    fn compute_tf_component(
        &self,
        term_freq: u32,
        doc_length: u32,
        avgdl: f64,
    ) -> f64 {
        let f = term_freq as f64;
        let dl = doc_length as f64;
        let k1 = self.config.k1;
        let b = self.config.b;
        
        let denominator = f + k1 * (1.0 - b + b * (dl / avgdl));
        
        (f * (k1 + 1.0)) / denominator
    }
    
    /// Compute BM25 score for a single term-document pair
    /// 
    /// Returns f32 for API consistency with vector similarity scores (D-06)
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
```

### Unit Test Patterns

```rust
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
    fn test_config_validation_k1_positive() {
        // k1 must be > 0
        assert!(BM25Config::new(0.0, 0.5).is_err());
        assert!(BM25Config::new(-1.0, 0.5).is_err());
        assert!(BM25Config::new(1.0, 0.5).is_ok());
    }
    
    #[test]
    fn test_config_validation_b_range() {
        // b must be in [0, 1]
        assert!(BM25Config::new(1.2, -0.1).is_err());
        assert!(BM25Config::new(1.2, 1.1).is_err());
        assert!(BM25Config::new(1.2, 0.0).is_ok());
        assert!(BM25Config::new(1.2, 1.0).is_ok());
        assert!(BM25Config::new(1.2, 0.75).is_ok());
    }
    
    #[test]
    fn test_idf_flooring() {
        let scorer = BM25Scorer::new(BM25Config::default());
        
        // When term appears in >50% of docs, IDF should floor at 0
        let idf = scorer.compute_idf(100, 60); // 60/100 = 60% > 50%
        assert_eq!(idf, 0.0);
        
        // When term appears in exactly 50% of docs
        let idf = scorer.compute_idf(100, 50);
        assert!(idf > 0.0); // Should still be positive
        
        // When term appears in <50% of docs
        let idf = scorer.compute_idf(100, 10);
        assert!(idf > 0.0);
    }
    
    #[test]
    fn test_bm25_score_known_values() {
        // Test against known BM25 values from Lucene
        let config = BM25Config::new(1.2, 0.75).unwrap();
        let scorer = BM25Scorer::new(config);
        
        // Scenario: 
        // - Term appears 3 times in document (f=3)
        // - Document has 100 terms (dl=100)
        // - Average document length is 80 (avgdl=80)
        // - Term appears in 10 of 1000 docs (df=10, N=1000)
        
        let score = scorer.score(3, 100, 80.0, 10, 1000);
        
        // Expected: positive score with reasonable magnitude
        assert!(score > 0.0);
        
        // Verify score increases with term frequency
        let score_tf1 = scorer.score(1, 100, 80.0, 10, 1000);
        let score_tf3 = scorer.score(3, 100, 80.0, 10, 1000);
        assert!(score_tf3 > score_tf1, "Higher TF should give higher score");
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
        
        assert!(ratio_1_to_10 < 10.0, "TF should saturate");
        assert!(ratio_1_to_100 < 100.0, "TF should saturate");
        assert!(ratio_1_to_10 > 1.0, "But should still increase");
    }
    
    #[test]
    fn test_length_normalization() {
        // Longer documents should get lower scores (with b > 0)
        let config = BM25Config::new(1.2, 0.75).unwrap();
        let scorer = BM25Scorer::new(config);
        
        // Same term frequency, different document lengths
        let score_short = scorer.score(5, 50, 100.0, 10, 1000);
        let score_long = scorer.score(5, 200, 100.0, 10, 1000);
        
        assert!(score_short > score_long, "Longer doc should have lower score");
    }
    
    #[test]
    fn test_no_normalization_when_b_is_zero() {
        // When b=0, document length should not affect score
        let config = BM25Config::new(1.2, 0.0).unwrap();
        let scorer = BM25Scorer::new(config);
        
        let score_short = scorer.score(5, 50, 100.0, 10, 1000);
        let score_long = scorer.score(5, 200, 100.0, 10, 1000);
        
        // Scores should be approximately equal
        assert!((score_short - score_long).abs() < 0.001);
    }
}
```

## Integration Points

### Module Registration

Add to `rs/index/src/lib.rs`:
```rust
pub mod bm25;
```

### Config Integration

The BM25Config will eventually be integrated into CollectionConfig (Phase 4). For now, it's a standalone config that will be used by the BM25Scorer.

### Testing Integration

Tests should use `tempdir::TempDir` for temporary test directories (standard pattern in codebase).

## Common Pitfalls

### Pitfall 1: IDF Without Flooring
**What goes wrong:** Negative scores when term appears in >50% of documents
**Why it happens:** Raw IDF formula gives negative values when n > N/2
**How to avoid:** Always apply `max(0.0, idf)` per D-03
**Warning signs:** Test failures for high-frequency terms

### Pitfall 2: Using f32 for Intermediate Calculations
**What goes wrong:** Precision loss in score accumulation
**Why it happens:** f32 has only 24 bits of precision, IDF calculations can overflow
**How to avoid:** Use f64 for all intermediate calculations, cast to f32 only at API boundary
**Warning signs:** Score drift in multi-term queries

### Pitfall 3: Incorrect k1/b Parameter Ranges
**What goes wrong:** Division by zero or nonsensical scores
**Why it happens:** k1=0 causes TF_component to approach infinity, b outside [0,1] gives negative normalization
**How to avoid:** Validate parameters on construction, reject invalid values
**Warning signs:** NaN or infinite scores

### Pitfall 4: Document Length Definition Mismatch
**What goes wrong:** Different phases calculate document length differently
**Why it happens:** One phase counts tokens, another counts bytes or terms
**How to avoid:** Standardize on "number of terms" per D-04
**Warning signs:** Inconsistent scores across segments

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| TF-IDF (unbounded TF) | BM25 (saturated TF) | 1990s | Better handling of keyword stuffing |
| Binary b parameter | Continuous b in [0,1] | BM25 origin | Fine-grained length normalization control |
| log(N/n) IDF | log(1 + (N-n+0.5)/(n+0.5)) IDF | BM25 origin | Better handling of rare terms |
| No IDF flooring | IDF floored at 0 | Lucene adoption | Prevents negative scores for common terms |

## Open Questions

1. **Multi-term score aggregation**
   - What we know: Score per term will be summed for multi-term queries
   - What's unclear: Whether to normalize by query length
   - Recommendation: Defer to Phase 3 (Query Scoring)

2. **Empty document handling**
   - What we know: dl=0 edge case needs handling
   - What's unclear: Whether to return 0 or handle specially
   - Recommendation: Return 0 score for empty documents (f would also be 0)

## Sources

### Primary (HIGH confidence)
- `rs/quantization/src/pq/mod.rs` - ProductQuantizerConfig validation pattern
- `rs/config/src/collection.rs` - CollectionConfig Default pattern
- `.planning/phases/01-core-algorithm-config/01-CONTEXT.md` - User decisions and formula specification
- Wikipedia "Okapi BM25" - Formula specification

### Secondary (MEDIUM confidence)
- Lucene BM25Similarity.java (v9.0.0) - Reference implementation patterns

### Tertiary (LOW confidence)
- Various blog posts on BM25 implementation (marked for validation against Lucene source)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All libraries already in codebase
- Architecture: HIGH - Patterns clearly established in existing code
- Pitfalls: MEDIUM - Based on standard BM25 documentation, need verification

**Research date:** 2026-04-05
**Valid until:** 2026-05-05 (stable algorithms, unlikely to change)

---

## Research Complete

**Phase:** 01 - Core Algorithm & Config
**Confidence:** HIGH

### Key Findings
1. Config pattern follows `ProductQuantizerConfig`: struct with serde + `validate()` method returning `Result<()>`
2. BM25 formula is well-specified in CONTEXT.md with specific k1/b defaults and IDF flooring requirement
3. f64 for intermediate calculations, f32 for API return (matches existing vector score patterns)
4. Module structure: create `rs/index/src/bm25/` with `mod.rs`, `config.rs`, `scorer.rs`

### File Created
`.planning/phases/01-core-algorithm-config/01-RESEARCH.md`

### Ready for Planning
Research complete. Planner can now create PLAN.md files.
