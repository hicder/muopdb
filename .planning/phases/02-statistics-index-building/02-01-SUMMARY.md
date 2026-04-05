---
phase: 02-statistics-index-building
plan: 01
type: summary
subsystem: bm25
tags: [bm25, statistics, index-building, core]
dependency_graph:
  requires: []
  provides: [STATS-01, STATS-02, STATS-03]
  affects: [bm25-scorer, mutable-segment]
tech_stack:
  added: []
  patterns:
    - "DashMap<u128, RwLock<UserBm25Stats>> - per-user concurrent statistics"
    - "HashMap-based TF and DF tracking"
    - "Document length (DL) tracking for BM25 avgdl calculation"
key_files:
  created:
    - rs/index/src/bm25/stats_collector.rs
  modified:
    - rs/index/src/bm25/mod.rs
decisions:
  - "Used same DashMap pattern as MultiTermBuilder for per-user isolation"
  - "Stored term_frequencies per document to support BM25 TF component"
  - "Tracked doc_frequencies (DF) at user level for IDF calculation"
  - "Stored total_term_count for efficient avgdl calculation at query time"
metrics:
  duration: ~15 minutes
  completed_date: 2025-01-05
  tasks_completed: 4
  tests_added: 9
---

# Phase 02 Plan 01: Statistics Collection Infrastructure Summary

## Overview

Implemented the BM25 statistics collection infrastructure that tracks document frequency (DF), term frequency (TF), and document lengths per user during document insertion. This provides the statistical foundation needed for BM25 scoring at query time.

## What Was Built

### Core Types

1. **`DocStats`** - Per-document statistics containing:
   - `term_count: u32` - Document length (dl) in terms
   - `term_frequencies: HashMap<String, u32>` - TF counts for each term in the document

2. **`UserBm25Stats`** - Per-user statistics container with:
   - `doc_stats: HashMap<u32, DocStats>` - All document statistics for the user
   - `doc_frequencies: HashMap<String, u32>` - DF counts (number of docs containing each term)
   - `total_term_count: u64` - Sum of all document lengths for avgdl calculation
   - Helper methods: `get_df()`, `get_tf()`, `avg_document_length()`, `num_documents()`

3. **`MultiBm25StatsCollector`** - Multi-user concurrent collector using:
   - `DashMap<u128, RwLock<UserBm25Stats>>` - Same pattern as MultiTermBuilder
   - Methods: `add_document()`, `get_stats_for_user()`, `get_all_user_ids()`, `has_user()`, `num_users()`

### Key Implementation Details

- **Pattern Consistency**: Follows the exact same DashMap + RwLock pattern as `MultiTermBuilder` for per-user data isolation
- **Concurrent Access**: DashMap enables lock-free reads across users, RwLock per user for document addition
- **Term Frequency Calculation**: `compute_term_frequencies()` helper counts token occurrences
- **DF Tracking**: Each unique term in a document increments the DF count
- **DL Tracking**: Total token count per document stored for BM25 length normalization

### Tests Added (9 total)

1. `test_single_user_single_doc` - Basic TF and DL tracking
2. `test_df_tracking_multiple_docs` - DF counts across documents
3. `test_multi_user_isolation` - Verify stats don't leak between users
4. `test_term_frequency_counting` - Repeated terms counted correctly
5. `test_total_term_count` - Total terms and avgdl calculation
6. `test_empty_document` - Empty doc handling (dl=0)
7. `test_get_all_user_ids` - User enumeration
8. `test_has_user_and_nonexistent_user` - User existence checking
9. `test_get_tf_helper` - TF lookup helper method

## Verification Results

```
running 9 tests
test bm25::stats_collector::tests::test_df_tracking_multiple_docs ... ok
test bm25::stats_collector::tests::test_empty_document ... ok
test bm25::stats_collector::tests::test_get_all_user_ids ... ok
test bm25::stats_collector::tests::test_get_tf_helper ... ok
test bm25::stats_collector::tests::test_has_user_and_nonexistent_user ... ok
test bm25::stats_collector::tests::test_multi_user_isolation ... ok
test bm25::stats_collector::tests::test_single_user_single_doc ... ok
test bm25::stats_collector::tests::test_term_frequency_counting ... ok
test bm25::stats_collector::tests::test_total_term_count ... ok

test result: ok. 9 passed; 0 failed; 0 ignored
```

- **Build**: `cargo build --release -p index` ✓
- **Clippy**: No warnings in stats_collector.rs ✓
- **Module exports**: `MultiBm25StatsCollector`, `UserBm25Stats`, `DocStats` public ✓

## Deviation from Plan

**None** - Plan executed exactly as written. All 4 tasks completed without issues.

## Integration Notes

The `MultiBm25StatsCollector` is designed to integrate with `MutableSegment`:

```rust
// In MutableSegment::insert_for_user():
self.bm25_stats_collector.add_document(
    user_id, 
    point_id, 
    tokenized_document_text
)?;
```

The statistics will be:
1. **Collected during insertion** via this collector
2. **Flushed to segment files** alongside the term index (next plan)
3. **Loaded by BM25Scorer** for IDF and score calculation
4. **Used in BM25 formula**: `score = IDF * (TF * (k1 + 1)) / (TF + k1 * (1 - b + b * dl/avgdl))`

## Next Steps

This infrastructure enables:
- Phase 02 Plan 02: Statistics persistence (flush stats to segment files)
- Phase 02 Plan 03: Query-time statistics loading for BM25Scorer
- Phase 03: BM25 + vector hybrid search integration

## Commit

```
6293326 feat(bm25): add statistics collection infrastructure
```
