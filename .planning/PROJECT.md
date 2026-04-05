# BM25 Support for MuopDB

## What This Is

Add BM25 (Best Matching 25) ranking algorithm to MuopDB's text search capabilities. BM25 is a probabilistic ranking function used by search engines to rank documents by relevance to a given search query. This will enable hybrid search combining BM25 text relevance scores with vector similarity scores.

The implementation will integrate with the existing `multi_terms` module, extending the `TermIndex` infrastructure to support document frequency statistics, term frequency saturation, and length normalization - the core components of BM25 scoring.

## Core Value

Users can search documents using BM25-ranked text relevance, enabling accurate keyword-based retrieval that complements vector similarity search for hybrid retrieval workflows.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] BM25 scoring algorithm implementation with configurable k1 and b parameters
- [ ] Document frequency statistics collection during index build
- [ ] Average document length calculation per user/collection
- [ ] Integration with existing TermIndex and MultiTermIndex infrastructure
- [ ] Scoring API returning BM25 scores for query-document pairs
- [ ] Support for combining BM25 scores with vector similarity scores (hybrid search)

### Out of Scope

- Full-text query language (e.g., AND/OR/NOT operators) — defer to future
- BM25F (field-weighted variant) — defer to future
- Real-time BM25 index updates — initial implementation for immutable segments only

## Context

MuopDB currently provides vector similarity search via SPANN indices. The `multi_terms` module (`rs/index/src/multi_terms/`) provides basic term-based indexing with posting lists, but lacks relevance ranking. Users storing AI memories with both text and embeddings need hybrid search to find relevant content.

Existing infrastructure:
- `TermIndex`: Per-user term-to-posting-list mapping
- `MultiTermIndex`: Lazy-loaded per-user term indices
- `MultiTermBuilder`: Builds term indices during segment construction
- Posting list iterators: Elias-Fano encoded posting lists

BM25 formula:
```
score(D, Q) = Σ IDF(qi) * (f(qi, D) * (k1 + 1)) / (f(qi, D) + k1 * (1 - b + b * |D| / avgdl))
```

Where:
- `f(qi, D)`: Term frequency of query term qi in document D
- `|D|`: Document length
- `avgdl`: Average document length in the collection
- `k1`: Term frequency saturation parameter (typically 1.2-2.0)
- `b`: Length normalization parameter (typically 0.75)
- `IDF(qi)`: Inverse document frequency = ln((N - n(qi) + 0.5) / (n(qi) + 0.5) + 1)

## Constraints

- **Architecture**: Must integrate with existing segment-based architecture (mutable/immutable segments)
- **Storage**: Statistics must persist with segment files for recovery
- **Concurrency**: Must work with existing DashMap-based multi-user indices
- **Performance**: Scoring should not significantly impact query latency

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Integrate with multi_terms module | Existing term infrastructure, minimal duplication | — Pending |
| Store statistics per-segment | Aligns with segment-based architecture | — Pending |

---
*Last updated: 2026-04-05 after initialization*

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state
