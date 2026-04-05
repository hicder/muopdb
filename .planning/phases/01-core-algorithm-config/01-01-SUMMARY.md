---
phase: 01-core-algorithm-config
plan: 01
subsystem: bm25
tags: [bm25, scoring, config, search]
dependency_graph:
  requires: []
  provides: [BM25Config, BM25Scorer]
  affects: [rs/index/src/multi_terms/]
tech-stack:
  added: []
  patterns:
    - Config struct with serde + validate() (from ProductQuantizerConfig)
    - Default trait for standard parameters
    - f64 intermediates, f32 API returns
key-files:
  created:
    - rs/index/src/bm25/mod.rs
    - rs/index/src/bm25/config.rs
    - rs/index/src/bm25/scorer.rs
  modified:
    - rs/index/src/lib.rs
key-decisions:
  - "IDF formula: ln(1 + (N-n+0.5)/(n+0.5)) with flooring at 0.0"
  - "Default parameters: k1=1.2, b=0.75 (Lucene/Elasticsearch defaults)"
  - "Parameter validation: k1 > 0, b in [0, 1]"
  - "f64 for intermediate calculations, f32 for API returns"
metrics:
  duration: 30
  completed_date: 2026-04-05
  test_count: 28
---

# Phase 01 Plan 01: Core BM25 Algorithm & Config Summary

## One-Liner
Implemented foundational BM25 scoring infrastructure with validated configuration (k1, b parameters) and complete scoring formula including IDF computation with flooring and TF saturation.

## What Was Built

### BM25Config (rs/index/src/bm25/config.rs)
- Configuration struct with `k1` (term frequency saturation) and `b` (length normalization) parameters
- Default implementation using Lucene/Elasticsearch defaults: k1=1.2, b=0.75
- Validation ensuring k1 > 0 and b ∈ [0, 1]
- Constructor `BM25Config::new(k1, b)` with built-in validation
- Serde serialization/deserialization support

### BM25Scorer (rs/index/src/bm25/scorer.rs)
- Scorer struct holding BM25Config
- `compute_idf(total_docs, doc_freq)`: IDF = ln(1 + (N-n+0.5)/(n+0.5)), floored at 0.0
- `compute_tf_component(term_freq, doc_length, avgdl)`: TF = (f×(k1+1)) / (f+k1×(1-b+b×(dl/avgdl)))
- `score()`: Returns f32 for API consistency, uses f64 for all intermediate calculations
- Comprehensive logging via `tracing::debug!`

### Module Registration
- New `rs/index/src/bm25/` module directory
- Exports: `BM25Config`, `BM25Scorer`
- Registered in `rs/index/src/lib.rs` as `pub mod bm25`

## Test Coverage

28 total tests across config and scorer:

**Config Tests (12):**
- Default config validation (k1=1.2, b=0.75)
- Custom parameter validation
- k1 boundary cases (zero, negative, positive)
- b boundary cases (-0.1, 0.0, 0.5, 1.0, 1.1)
- Serde serialization/deserialization

**Scorer Tests (16):**
- IDF computation with known values
- IDF flooring behavior at high document frequency
- TF saturation (10x TF increase < 10x score increase)
- Length normalization (b > 0: longer docs score less, b = 0: equal scores)
- Score increases with term frequency
- Score decreases with document frequency
- Empty document handling
- Custom k1 parameter effects

## API Usage

```rust
use index::bm25::{BM25Config, BM25Scorer};

// Default configuration
let config = BM25Config::default();
// Or custom with validation
let config = BM25Config::new(1.5, 0.5)?;

// Create scorer
let scorer = BM25Scorer::new(config);

// Compute BM25 score
let score = scorer.score(
    3,        // term frequency in document
    100,      // document length (terms)
    80.0,     // average document length
    10,       // document frequency (docs containing term)
    1000,     // total documents
);
```

## Deviations from Plan

### Auto-corrected Issues

**1. [Rule 1 - Test Fix] IDF Flooring Test Expectations**
- **Found during:** Task 2 test implementation
- **Issue:** Original test expected IDF to floor at 0 when term appears in >50% of docs, but with the `ln(1 + ...)` formula, IDF is always positive until n approaches N (100%)
- **Fix:** Updated tests to verify actual formula behavior - IDF approaches 0 as document frequency approaches total documents
- **Commit:** Included in Task 2 commit (9ddbd9e)

**2. [Rule 1 - Code Fix] compute_idf variable naming bug**
- **Found during:** Initial scorer.rs implementation
- **Issue:** Accidentally redeclared `n` variable multiple times, causing formula error
- **Fix:** Used distinct variable names: `n` for doc_freq, `n_docs` for total_docs
- **Commit:** Included in Task 2 commit (9ddbd9e)

### No Other Deviations

All other plan requirements implemented exactly as specified:
- ✅ BM25Config with serde, Default, validation per D-01, D-02
- ✅ BM25Scorer with IDF computation per D-03, D-04
- ✅ f64 intermediate calculations per D-05
- ✅ f32 API return per D-06
- ✅ Module registration in lib.rs

## Self-Check

- [x] `rs/index/src/bm25/config.rs` exists with BM25Config struct
- [x] `rs/index/src/bm25/scorer.rs` exists with BM25Scorer struct
- [x] `rs/index/src/bm25/mod.rs` exists with exports
- [x] `rs/index/src/lib.rs` contains `pub mod bm25`
- [x] 28 tests pass (`cargo test -p index bm25::`)
- [x] Code compiles (`cargo check -p index`)

**Result:** PASSED

## Commits

1. `449572f`: feat(bm25): implement BM25Config with validation
2. `9ddbd9e`: feat(bm25): implement BM25Scorer with scoring formula

## Next Steps

This plan provides the foundational BM25 scoring infrastructure. Subsequent phases will:
- Collect term statistics (document frequency, term frequency) during indexing
- Store per-segment BM25 statistics alongside segment files
- Integrate BM25Scorer with MultiTermIndex for query-time scoring
- Implement hybrid search combining BM25 text relevance with vector similarity
