---
phase: 04-hybrid-search-integration
plan: 01
type: summary
subsystem: bm25
tags: [hybrid-search, score-normalization, fusion, bm25]
requirements:
  - HYB-01
  - HYB-02
key-decisions:
  - "Min-max normalization to [0,1] is the default approach for unifying BM25 and vector score scales"
  - "NaN values are filtered during normalization to prevent score contamination"
  - "Alpha parameter controls fusion weighting: 1.0=text-only, 0.0=vector-only"
  - "merge_results handles missing scores by assigning 0.0 (neutral contribution)"
  - "HybridCandidate ordering uses hybrid_score descending with doc_id tie-breaking"
dependency-graph:
  requires: []
  provides: [score-normalization-api, hybrid-fusion-api]
  affects: [hybrid-searcher]
tech-stack:
  added: []
  patterns: [min-max-normalization, linear-combination-fusion, score-statistics]
metrics:
  files-created: 2
  files-modified: 1
  tests-added: 35
  lines-of-code: ~600
  duration: 30m
---

# Phase 04 Plan 01: Score Normalization and Fusion Summary

## Overview

Implemented the mathematical foundation for hybrid search by creating score normalization utilities and fusion algorithms that combine BM25 text relevance with vector similarity scores.

## Implementation

### 1. ScoreNormalizer (`rs/index/src/bm25/normalizer.rs`)

A utility for normalizing BM25 scores (unbounded, typically 0-20+) to the [0,1] range used by vector similarity scores.

**Key Features:**
- `normalize_scores(scores: &[f32]) -> Vec<f32>` - Min-max normalization
- `batch_normalize(score_lists: &[Vec<f32>]) -> Vec<Vec<f32>>` - Batch processing for multiple segments
- `normalize_with_clipping(scores: &[f32], std_devs: f32)` - Outlier-aware normalization
- `score_stats(scores: &[f32]) -> ScoreStats` - Statistics for diagnostics

**Edge Cases Handled:**
- Empty input → returns empty vector
- Single value → returns [1.0]
- All identical values → returns all [0.0] (no variance)
- NaN values → filtered before processing
- Negative values → normalized correctly to [0,1]

### 2. HybridScoreFusion (`rs/index/src/bm25/fusion.rs`)

A fusion engine for combining BM25 and vector search results using linear combination.

**Core Formula:**
```
hybrid_score = alpha * bm25_score + (1 - alpha) * vector_score
```

**Key Features:**
- `FusionConfig` - Configurable alpha parameter with clamping to [0,1]
- `combine(bm25_score, vector_score)` - Linear combination with stored alpha
- `weighted_combine(bm25_score, vector_score, alpha)` - Per-query alpha override
- `merge_results(bm25_results, vector_results)` - Merge by doc_id with 0.0 for missing scores
- `fuse_aligned_results(docs)` - Process pre-aligned results

**HybridCandidate:**
- `doc_id: u128` - Document identifier
- `bm25_score: f32` - Text relevance score
- `vector_score: f32` - Vector similarity score  
- `hybrid_score: f32` - Combined score

**Ordering:** Descending by hybrid_score, ascending doc_id for ties

## Test Coverage

### Normalizer Tests (16 tests)
1. Basic normalization [1.0, 2.0, 3.0, 4.0, 5.0] → [0.0, 0.25, 0.5, 0.75, 1.0]
2. Single value returns [1.0]
3. All identical values return [0.0, 0.0, 0.0]
4. Empty input returns []
5. Negative values handled correctly
6. NaN values filtered
7. Batch normalization for multiple lists
8. Score statistics computation
9. Outlier clipping normalization
10. Range verification

### Fusion Tests (19 tests)
1. Equal weights (alpha=0.5) → average
2. Text-only (alpha=1.0) → BM25 only
3. Vector-only (alpha=0.0) → vector only
4. Zero BM25 score handling
5. Alpha clamping (>1 → 1, <0 → 0)
6. Result merging from both sources
7. BM25-only results
8. Vector-only results
9. Empty result handling
10. Tie-breaking by doc_id
11. NaN filtering in merge_results
12. Per-query weighted_combine
13. Aligned result fusion

## Verification

```bash
# All BM25 tests pass (122 tests)
cargo test --release -p index -- bm25

# No new clippy warnings in new files
cargo clippy -p index
```

## Integration Points

**From bm25/normalizer.rs:**
- `normalize_scores()` → Used by hybrid searcher to normalize BM25 results before fusion

**From bm25/fusion.rs:**
- `HybridScoreFusion::combine()` → Used by collection search to compute final rankings
- `merge_results()` → Used when combining BM25 and vector search results

## Files

| File | Lines | Purpose |
|------|-------|---------|
| `rs/index/src/bm25/normalizer.rs` | 410 | Score normalization utilities |
| `rs/index/src/bm25/fusion.rs` | 578 | Hybrid score fusion algorithms |
| `rs/index/src/bm25/mod.rs` | 19 | Module exports (updated) |

## Deviations from Plan

None - plan executed exactly as written.

## Known Stubs

None - all functionality is fully implemented with proper data flow.

## Decisions Made

1. **Min-Max vs Z-Score Normalization:** Chose min-max for [0,1] output to match vector similarity scale. Z-score would require additional shifting/scaling.

2. **NaN Handling:** Filter NaN values during normalization to prevent score contamination. Documents with NaN scores are excluded from results.

3. **Missing Score Strategy:** In `merge_results()`, documents present in only one result set get 0.0 for the missing score type. This is neutral in the linear combination.

4. **Alpha Clamping:** Alpha values outside [0,1] are clamped to the nearest bound in the config constructor, ensuring valid fusion weighting.

5. **Outlier Clipping:** Optional 3-sigma clipping available for BM25 scores that may have extreme spikes. Default normalization doesn't clip.

## Next Steps

These components provide the mathematical foundation for hybrid search. The next phase will integrate them into:
- Collection-level hybrid search API
- gRPC endpoints for hybrid queries  
- Score caching for repeated queries
- Multi-segment fusion with segment-level normalization

## Commit

```
feat(bm25): score normalization and hybrid fusion algorithms

Hash: 3c012d8
Files: rs/index/src/bm25/fusion.rs, rs/index/src/bm25/normalizer.rs, rs/index/src/bm25/mod.rs
```
