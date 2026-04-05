# Roadmap: BM25 Support for MuopDB

## Overview

This roadmap delivers BM25 text ranking for MuopDB, enabling hybrid search combining text relevance with vector similarity. The journey progresses from core scoring algorithm implementation through statistics collection, query scoring, and finally hybrid search integration—each phase building on the previous to deliver incremental value.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Core Algorithm & Config** - BM25 scoring formula with configurable parameters
- [x] **Phase 2: Statistics & Index Building** - Collect and persist BM25 statistics during indexing (completed 2026-04-05)
- [ ] **Phase 3: Query Scoring** - Text queries return BM25-ranked results
- [ ] **Phase 4: Hybrid Search Integration** - Combined text+vector relevance search

## Phase Details

### Phase 1: Core Algorithm & Config
**Goal**: BM25 scoring formula implementation with configurable parameters validated by unit tests
**Depends on**: Nothing (first phase)
**Requirements**: CORE-01, CORE-02, CORE-03, CORE-04, API-01
**Success Criteria** (what must be TRUE):
  1. BM25Config struct accepts k1 and b parameters with validation (k1 > 0, 0 <= b <= 1)
  2. BM25 scoring formula computes correct scores for given query-document pairs
  3. IDF computation floors at 0 to prevent negative scores for high-frequency terms
  4. Unit tests verify scoring matches expected BM25 behavior from Lucene reference
**Plans**: 2 plans

Plans:
- [x] 01-01-PLAN.md — BM25Config and BM25Scorer implementation
- [x] 01-02-PLAN.md — Comprehensive unit tests for config and scorer

### Phase 2: Statistics & Index Building
**Goal**: Index building collects and persists BM25 statistics per segment for query-time scoring
**Depends on**: Phase 1
**Requirements**: STAT-01, STAT-02, STAT-03, STAT-04, STAT-05, BLD-01, BLD-02, BLD-03, BLD-04, BLD-05
**Success Criteria** (what must be TRUE):
  1. Document frequency statistics are collected per term during index build
  2. Document lengths are tracked and stored per document
  3. Average document length is computed and stored per segment
  4. Statistics are serialized with rkyv and stored in bm25_stats/ subdirectory
  5. Tokenization pipeline produces stemmed tokens from text documents
**Plans**: 3 plans (Wave 1: statistics collection + tokenization, Wave 2: persistence + integration)

Plans:
- [x] 02-01-PLAN.md — BM25 statistics collection infrastructure (MultiBm25StatsCollector, DF/TF/DL tracking)
- [x] 02-02-PLAN.md — Tokenization pipeline (UnicodeTokenizer, StopWordFilter, TokenizerPipeline)
- [x] 02-03-PLAN.md — Statistics persistence (rkyv serialization, Bm25StatsWriter/Reader, MutableSegment integration)

### Phase 3: Query Scoring
**Goal**: Users can execute text queries and receive BM25-ranked results
**Depends on**: Phase 2
**Requirements**: QRY-01, QRY-02, QRY-03, QRY-04, API-02
**Success Criteria** (what must be TRUE):
  1. Single-term queries return documents ranked by BM25 score
  2. Multi-term queries aggregate scores across query terms correctly
  3. Top-K retrieval returns highest-scoring K documents for text query
  4. Cross-segment queries aggregate scores from multiple segments
  5. BM25Scorer is integrated with existing MultiTermIndex
**Plans**: 4 plans (Wave 1: core search + top-K, Wave 2: cross-segment + integration)

Plans:
- [ ] 03-01-PLAN.md — BM25Searcher component with single/multi-term query scoring
- [ ] 03-02-PLAN.md — Top-K retrieval with min-heap selection (O(N log K))
- [ ] 03-03-PLAN.md — Cross-segment aggregation and Collection::search_text()
- [ ] 03-04-PLAN.md — Integration tests and API verification (QRY-01 through QRY-04, API-02)

### Phase 4: Hybrid Search Integration
**Goal**: Users can search with combined BM25 text relevance and vector similarity scores
**Depends on**: Phase 3
**Requirements**: HYB-01, HYB-02, HYB-03, HYB-04, API-03, API-04
**Success Criteria** (what must be TRUE):
  1. BM25 scores are normalized to [0,1] range for hybrid fusion
  2. Users can configure alpha parameter for linear combination fusion
  3. Hybrid queries return documents ranked by combined score
  4. Per-user isolation is maintained in hybrid search results
  5. gRPC API supports text-only, vector-only, and hybrid query modes
**Plans**: 3 plans (Wave 1: normalization + fusion, Wave 2: hybrid searcher, Wave 3: API + tests)

Plans:
- [ ] 04-01-PLAN.md — Score normalization and fusion algorithms (HYB-01, HYB-02)
- [ ] 04-02-PLAN.md — Hybrid searcher and Collection::search_hybrid() (HYB-03, HYB-04)
- [ ] 04-03-PLAN.md — gRPC API extensions and integration tests (API-03, API-04)

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Core Algorithm & Config | 0/TBD | Not started | - |
| 2. Statistics & Index Building | 3/3 | Complete   | 2026-04-05 |
| 3. Query Scoring | 0/TBD | Not started | - |
| 4. Hybrid Search Integration | 0/TBD | Not started | - |

---
*Roadmap created: 2026-04-05*
