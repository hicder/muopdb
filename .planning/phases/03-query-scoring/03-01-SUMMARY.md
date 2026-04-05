# Phase 03 Plan 01: BM25 Searcher Core - Summary

---
phase: 03-query-scoring
plan: 01
subsystem: bm25
tags: [bm25, search, query, scoring]
duration: 45min
completed: 2026-04-05
---

## One-Liner Summary

Implemented the core `Bm25Searcher` component with single-term (QRY-01) and multi-term (QRY-02) query scoring capabilities, integrated with `MultiTermIndex` and `Bm25StatsReader`.

## What Was Built

### Core Components

1. **`Bm25Searcher`** (`rs/index/src/bm25/searcher.rs`)
   - Main search implementation with `search()` method
   - Single-term scoring via `score_single_term()` (QRY-01)
   - Multi-term scoring with score aggregation via `score_multi_term()` (QRY-02)
   - Uses `TokenizerPipeline` for consistent query tokenization
   - Integrates with `MultiTermIndex` for posting list access
   - Uses `Bm25StatsReader` for document statistics (IDF, doc length)

2. **Query Types** (`rs/index/src/bm25/query.rs`)
   - `Bm25Query`: Search query with text, user_id, max_results
   - `Bm25SearchResult`: Individual result with point_id and score
   - `Bm25SearchResults`: Container with sorted results and total_hits
   - Proper ordering traits for descending score sort

### Integration Points

- **Tokenizer**: Uses `TokenizerPipeline::english()` for query tokenization (same as indexing)
- **Stats Reader**: `Bm25StatsReader` provides per-user statistics (num_docs, avgdl, term DF, doc lengths)
- **Term Index**: `MultiTermIndex` provides posting list iteration via `get_posting_list_iterator()`
- **Scorer**: `BM25Scorer` computes BM25 scores with IDF flooring at 0

## Verification Results

### Tests Passing

```
running 8 tests
test bm25::searcher::tests::test_empty_query ... ok
test bm25::searcher::tests::test_max_results_limiting ... ok
test bm25::searcher::tests::test_multi_term_score_aggregation ... ok
test bm25::searcher::tests::test_query_only_stop_words ... ok
test bm25::searcher::tests::test_query_tokenization ... ok
test bm25::searcher::tests::test_single_term_search ... ok
test bm25::searcher::tests::test_unknown_term ... ok
test bm25::searcher::tests::test_unknown_user ... ok

test result: ok. 8 passed; 0 failed
```

### All BM25 Tests

```
test result: ok. 84 passed; 0 failed; 0 ignored
```

## Key Implementation Details

### Single-Term Scoring (QRY-01)

```rust
fn score_single_term(&self, user_id: u128, term: &str, stats: &UserBm25StatsView) 
    -> Result<HashMap<u32, f32>> {
    // 1. Look up term ID from MultiTermIndex
    // 2. Get posting list iterator
    // 3. For each document in posting list:
    //    - Get doc_length from stats
    //    - Get doc_freq from stats (for IDF)
    //    - Call BM25Scorer::score() to compute score
}
```

### Multi-Term Scoring (QRY-02)

```rust
fn score_multi_term(&self, user_id: u128, terms: &[String], stats: &UserBm25StatsView) 
    -> Result<HashMap<u32, f32>> {
    // 1. For each term, call score_single_term()
    // 2. Aggregate scores by summing per document
    // 3. Documents matching multiple terms get higher scores
}
```

### BM25 IDF Behavior

- Terms appearing in >50% of documents get IDF floored to 0 (per D-03)
- Tests use 5 documents with "date" appearing in 2 docs (40% < 50%, positive IDF)
- This ensures non-zero scores for meaningful test verification

## Files Created/Modified

| File | Lines | Purpose |
|------|-------|---------|
| `rs/index/src/bm25/searcher.rs` | ~350 | Main Bm25Searcher implementation |
| `rs/index/src/bm25/query.rs` | ~270 | Query and result types |
| `rs/index/src/bm25/mod.rs` | ~17 | Module exports (already existed) |

## Technical Decisions

1. **Score Aggregation**: Multi-term queries sum scores across terms (standard BM25 behavior)
2. **Result Ordering**: Descending by score, ascending by point_id for ties
3. **Empty Query Handling**: Returns empty results (no error)
4. **Unknown Terms**: Gracefully returns empty results (no error)
5. **Unknown Users**: Returns error with descriptive message

## Deviation Documentation

None - plan executed as written.

## Checklist

- [x] `Bm25Query`, `Bm25SearchResult`, `Bm25SearchResults` types exist
- [x] `Bm25Searcher` struct with `search()` method
- [x] Single-term queries compute BM25 scores (QRY-01)
- [x] Multi-term queries aggregate scores (QRY-02)
- [x] Query tokenization uses same pipeline as indexing
- [x] All unit tests pass
- [x] Code compiles without errors
