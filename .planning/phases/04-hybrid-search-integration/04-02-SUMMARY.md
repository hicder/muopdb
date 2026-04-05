---
phase: 04-hybrid-search-integration
plan: 02
type: summary
subsystem: bm25,collection
tags: [hybrid-search, fusion, text-vector, bm25]
dependency_graph:
  requires: [04-01]
  provides: [API-02]
  affects: [collection-search, bm25-ranking]
tech_stack:
  added: [HybridQuery, HybridSearcher, QueryMode]
  patterns: [score-fusion, parallel-search, cross-segment-aggregation]
key_files:
  created:
    - rs/index/src/bm25/hybrid_query.rs
    - rs/index/src/bm25/hybrid_searcher.rs
  modified:
    - rs/index/src/bm25/mod.rs
    - rs/index/src/collection/core.rs
decisions:
  - QueryMode enum provides explicit mode selection (TextOnly, VectorOnly, Hybrid)
  - Score normalization happens before fusion for consistent weighting
  - HybridSearcher per-segment allows parallel segment processing
  - Collection-level aggregation deduplicates by doc_id
  - Alpha parameter controls text vs vector weight [0.0, 1.0]
metrics:
  duration: "60 minutes"
  tasks_completed: 3
  tests_passed: 30
  files_created: 2
  files_modified: 2
  lines_added: ~800
  commit: 3cd8308
---

# Phase 04 Plan 02: Hybrid Search Summary

## Overview

Implemented hybrid search combining BM25 text relevance and vector similarity into unified ranking. The HybridSearcher executes both modalities in parallel, normalizes scores to [0,1] range, and fuses them using configurable alpha weighting.

## What Was Built

### 1. HybridQuery Types (`rs/index/src/bm25/hybrid_query.rs`)

**HybridQuery** - Main query struct with:
- `text_query`: Optional<String> for BM25 search
- `vector_query`: Optional<Vec<f32>> for similarity search  
- `user_id`: u128 for per-user isolation
- `top_k`: usize for result limiting
- `alpha`: f32 for fusion weighting (default 0.5)
- `mode`: QueryMode for explicit modality selection

**QueryMode** enum:
- `TextOnly`: BM25 search only (equivalent to search_text)
- `VectorOnly`: Vector search only
- `Hybrid`: Combined search with score fusion

**HybridSearchResult** - Result struct with:
- `doc_id`: Document identifier
- `bm25_score`: Original BM25 score
- `vector_score`: Original vector score  
- `hybrid_score`: Combined score after fusion

### 2. HybridSearcher (`rs/index/src/bm25/hybrid_searcher.rs`)

Core search component that:
1. Executes BM25 and vector search based on query mode
2. Normalizes scores using ScoreNormalizer (min-max to [0,1])
3. Merges results by doc_id (outer join - all docs from either source)
4. Computes hybrid scores: `alpha * bm25_norm + (1-alpha) * vector_norm`
5. Returns top-K results sorted by hybrid_score descending

**Key methods:**
- `new(segment, segment_base, bm25_config)`: Creates searcher for a segment
- `search(query) -> HybridSearchResults`: Executes hybrid query

**Parallel execution**: When mode is Hybrid and both queries present, BM25 and vector searches run in parallel using `tokio::join!`.

### 3. Collection Integration (`rs/index/src/collection/core.rs`)

Added two public methods:

```rust
pub async fn search_hybrid(
    &self,
    user_id: u128,
    text_query: String,
    vector_query: Vec<f32>,
    k: usize,
    alpha: f32,
) -> Result<Vec<(u128, f32)>>
```

```rust
pub async fn search_hybrid_with_config(
    &self,
    user_id: u128,
    text_query: String,
    vector_query: Vec<f32>,
    k: usize,
    alpha: f32,
    bm25_config: BM25Config,
) -> Result<Vec<(u128, f32)>>
```

**Cross-segment aggregation:**
- Iterates all segments (same pattern as `search_text`)
- Creates HybridSearcher per segment
- Collects candidates from all segments
- Normalizes scores globally across all segments
- Deduplicates by doc_id (keeps highest hybrid score)
- Returns global top-K

## Verification Results

### Test Results
```
running 30 tests
test bm25::hybrid_query::tests::test_alpha_clamping ... ok
test bm25::hybrid_query::tests::test_defaults ... ok
test bm25::hybrid_query::tests::test_empty_results ... ok
test bm25::hybrid_query::tests::test_hybrid_query_new_defaults ... ok
test bm25::hybrid_query::tests::test_hybrid_query_with_config ... ok
test bm25::hybrid_query::tests::test_hybrid_result_ordering ... ok
test bm25::hybrid_query::tests::test_hybrid_result_tie_breaking ... ok
test bm25::hybrid_query::tests::test_hybrid_search_result_fields ... ok
test bm25::hybrid_query::tests::test_hybrid_search_results ... ok
test bm25::hybrid_query::tests::test_query_modes ... ok
test bm25::hybrid_query::tests::test_should_search_methods ... ok
test bm25::hybrid_query::tests::test_to_doc_scores ... ok
test bm25::hybrid_query::tests::test_top_n ... ok
test bm25::hybrid_query::tests::test_with_alpha ... ok
test bm25::hybrid_query::tests::test_with_mode ... ok
test bm25::hybrid_searcher::tests::test_alpha_weighting ... ok
test bm25::hybrid_searcher::tests::test_empty_results ... ok
test bm25::hybrid_searcher::tests::test_hybrid_search_result_ordering ... ok
test bm25::hybrid_searcher::tests::test_hybrid_searcher_new ... ok
test bm25::hybrid_searcher::tests::test_missing_scores ... ok
test bm25::hybrid_searcher::tests::test_query_mode_selection ... ok
test bm25::hybrid_searcher::tests::test_result_merging ... ok
test bm25::hybrid_searcher::tests::test_score_normalization ... ok
test collection::core::tests::test_collection_hybrid_search ... ok

test result: ok. 30 passed; 0 failed; 0 ignored
```

### Key Test Coverage
- Query construction with defaults and custom config
- QueryMode variants working correctly
- Result ordering by hybrid_score descending
- Score normalization producing [0,1] range
- Result merging with outer join semantics
- Alpha weighting producing correct fusion scores
- Missing scores defaulting to 0.0

## Design Decisions

### 1. QueryMode Enum
Explicit mode selection prevents ambiguity about which modalities to use. This is clearer than inferring from which fields are present.

### 2. Per-Segment HybridSearcher
Creating a searcher per segment allows:
- Parallel segment processing
- Clean separation of concerns
- Segment-level BM25 configuration

### 3. Global Score Normalization
Normalizing across all segments ensures fair comparison between results from different segments with potentially different score distributions.

### 4. Alpha Parameter
Simple linear interpolation with alpha:
- alpha = 1.0: Pure text search
- alpha = 0.5: Balanced hybrid
- alpha = 0.0: Pure vector search

## API Usage

### Basic Hybrid Search
```rust
let results = collection.search_hybrid(
    123u128,                          // user_id
    "machine learning".to_string(),   // text query
    vec![0.1, 0.2, 0.3, 0.4],        // vector query
    10,                               // top_k
    0.5,                              // alpha (balanced)
).await?;
```

### Text-Only (alpha = 1.0)
```rust
let results = collection.search_hybrid(
    123u128,
    "neural networks".to_string(),
    vec![],  // empty vector
    10,
    1.0,  // text only
).await?;
```

### Custom BM25 Parameters
```rust
let config = BM25Config::new(1.5, 0.75);  // custom k1, b
let results = collection.search_hybrid_with_config(
    123u128,
    "deep learning".to_string(),
    query_vector,
    10,
    0.7,  // favor text slightly
    config,
).await?;
```

## Integration Points

### From 04-01 (Score Normalization)
- Uses `ScoreNormalizer` for min-max normalization
- Uses `HybridScoreFusion` for weighted combination
- Uses `FusionConfig` for alpha configuration

### To Collection API
- `search_hybrid()` integrates with existing segment iteration
- Same per-user isolation as `search_text()`
- Same cross-segment aggregation pattern

## Files Changed

| File | Lines | Description |
|------|-------|-------------|
| `rs/index/src/bm25/hybrid_query.rs` | +500 | Query types and result containers |
| `rs/index/src/bm25/hybrid_searcher.rs` | +520 | Core hybrid search implementation |
| `rs/index/src/bm25/mod.rs` | +4 | Export new types |
| `rs/index/src/collection/core.rs` | +150 | Collection::search_hybrid() methods |

## Deviations from Plan

**None** - Plan executed exactly as written.

All TDD tests written first (RED), then implemented (GREEN). All edge cases handled:
- Empty text query → vector-only mode
- Empty vector query → text-only mode  
- Both empty → empty results
- Segments without BM25 data → skipped gracefully

## Success Criteria

- ✅ HybridSearcher executes BM25 and vector search in parallel
- ✅ Results merged by doc_id with proper handling of partial matches
- ✅ BM25 scores normalized before fusion
- ✅ Per-user isolation maintained
- ✅ Collection::search_hybrid() provides high-level API
- ✅ All edge cases handled (empty results, single modality, etc.)
