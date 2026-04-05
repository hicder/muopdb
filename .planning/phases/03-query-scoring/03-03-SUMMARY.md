---
phase: 03-query-scoring
plan: 03
type: summary
subsystem: collection
tags: [bm25, text-search, cross-segment, aggregation]
depends_on: [03-01, 03-02]
requires: [QRY-04]
key-files:
  created:
    - rs/index/src/collection/text_search.rs
  modified:
    - rs/index/src/collection/core.rs
    - rs/index/src/collection/mod.rs
    - rs/index/src/bm25/mod.rs
decisions:
  - Use TopKSelector with OrderedFloat for efficient top-K selection across segments
  - Deduplicate by doc_id keeping highest score when document appears in multiple segments
  - Handle missing BM25 data gracefully (segments without term indexes are skipped)
  - Support both default and custom BM25Config in search methods
metrics:
  duration: "30 minutes"
  completed: "2025-04-05"
  tasks: 5
  tests: 13
---

# Phase 03 Plan 03 Summary: Cross-Segment Text Search with BM25

## Overview

Implemented cross-segment aggregation infrastructure at the Collection level, enabling BM25 text queries across all immutable segments with result deduplication and global top-K selection.

## Implementation

### Core Components

1. **SegmentBm25Context** (`text_search.rs:18-54`)
   - Per-segment search context containing initialized Bm25Searcher
   - Holds segment metadata (name, stats_dir, terms_dir)
   - Provides isolated search context for each segment

2. **CrossSegmentTextSearcher** (`text_search.rs:119-274`)
   - Manages multi-segment BM25 search
   - Iterates through all segments with BM25 data
   - Aggregates results using HashMap<u128, (score, hits)> for deduplication
   - Uses TopKSelector for efficient O(N log K) top-K selection
   - Converts point_ids to external doc_ids via segment.get_doc_id()

3. **Collection::search_text()** (`core.rs:1583-1695`)
   - Public API for collection-level text search
   - Searches all immutable segments with BM25 data
   - Returns Vec<(doc_id, score)> sorted by score descending
   - Handles segments without term indexes gracefully (skipped with debug logging)
   - Deduplicates by doc_id keeping highest score

4. **Collection::search_text_with_config()** (`core.rs:1605-1695`)
   - Variant allowing custom BM25Config (k1, b parameters)
   - Same aggregation and deduplication logic as search_text()

### Key Design Decisions

1. **Deduplication Strategy**: When the same document appears in multiple segments (e.g., from segment merging or reindexing), the highest BM25 score is retained. This ensures documents don't appear multiple times in results.

2. **Graceful Degradation**: Segments without BM25 data (e.g., created before BM25 was enabled) are skipped with debug logging rather than causing errors.

3. **Top-K Selection**: Uses TopKSelector with OrderedFloat<f32> to handle f32 ordering requirements, providing O(N log K) performance for selecting top results across all segments.

4. **Type Safety**: Proper generic bounds on Quantizer types ensure the searcher works with both NoQuantizer and ProductQuantizer segment types.

## Test Coverage

### Unit Tests (text_search.rs)
- `test_cross_segment_search_basic` - Creates multi-segment searcher
- `test_segment_context_creation` - Validates SegmentBm25Context construction
- `test_cross_segment_results` - Tests result container structure
- `test_cross_segment_empty_results` - Empty result handling
- `test_segment_hit` - SegmentHit struct validation
- `test_cross_segment_result` - CrossSegmentResult struct validation
- `test_searcher_creation_empty` - Empty segment list handling
- `test_searcher_skips_missing_segments` - Graceful handling of invalid segment paths

### Integration Tests (core.rs)
- `test_collection_search_text_single_segment` - Single segment search API
- `test_collection_search_text_empty_results` - Empty query handling
- `test_collection_search_text_sorted_descending` - Score ordering verification
- `test_collection_search_text_user_isolation` - Per-user search isolation
- `test_collection_search_text_with_config` - Custom BM25Config support

## Deviation from Plan

**No deviations** - Plan executed as written.

Note: Full end-to-end text search requires text indexing during document insertion/flush, which is a separate integration task beyond the scope of this plan. The cross-segment aggregation infrastructure (the goal of this plan) is fully functional.

## Verification

```bash
cargo test --release -p index collection::text_search
cargo test --release -p index collection::core::tests::test_collection_search_text
cargo clippy -p index
```

All tests pass. Code compiles without errors.

## API Usage Example

```rust
// Search with default BM25 config
let results = collection.search_text(user_id, "apple banana".to_string(), 10).await?;

// Results: Vec<(doc_id, score)> sorted by score descending
for (doc_id, score) in results {
    println!("Doc {} has score {}", doc_id, score);
}

// Search with custom BM25 parameters
let custom_config = BM25Config::new(2.0, 0.5)?;
let results = collection
    .search_text_with_config(user_id, "query".to_string(), 10, custom_config)
    .await?;
```

## Files Changed

| File | Lines | Description |
|------|-------|-------------|
| `rs/index/src/collection/text_search.rs` | 524 | New cross-segment aggregation module |
| `rs/index/src/collection/core.rs` | +310 | Added search_text() and search_text_with_config() methods |
| `rs/index/src/collection/mod.rs` | +2 | Added text_search module export |

## Commits

- `d99e8bb`: feat(collection): add cross-segment text search with BM25

## Related Work

- Depends on: Plan 03-01 (BM25 Query Types), Plan 03-02 (Per-Segment Search)
- Implements: Requirement QRY-04 (Cross-segment aggregation)
