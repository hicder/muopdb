# Project Research Summary

**Project:** MuopDB BM25 Text Ranking Support
**Domain:** Hybrid vector + text search for multi-user vector database
**Researched:** 2025-04-05
**Confidence:** HIGH

## Executive Summary

MuopDB is adding BM25 text ranking to enable hybrid search combining text relevance with vector similarity. Industry practice (Elasticsearch, Lucene, Tantivy) shows BM25 is the standard text scoring algorithm with well-understood implementation patterns. The recommended approach is a **custom BM25 scoring layer** integrated with MuopDB's existing `multi_terms` infrastructure rather than adopting a full search engine like Tantivy, which would create architectural impedance mismatch with MuopDB's segment-based storage model.

The implementation follows a three-tier statistics architecture aligned with MuopDB's segments: collection-level (document count, avgdl), term-level (document frequency, total TF), and document-level (field lengths/norms). Critical risks include segment-level statistics skew (scores not comparable across segments), hybrid search score normalization (BM25 and vector scores have different ranges), and floating-point precision loss in TF saturation calculations. These are mitigated by storing per-segment statistics with cross-segment aggregation at query time, implementing score normalization before fusion, and using precision-safe formula rewrites.

## Key Findings

### Recommended Stack

Custom BM25 implementation is preferred over integrating Tantivy due to MuopDB's unique segment architecture and mmap'd storage model. The BM25 formula itself is well-defined and straightforward to implement correctly, while external engines assume different document models and storage patterns.

**Core technologies:**
- **unicode-segmentation (v1.13.2)** — Unicode-compliant word tokenization — industry standard, zero-allocation iterators
- **rust-stemmers (v1.2.0)** — Snowball stemming — already in codebase, 17 language support
- **stop-words (v0.10.0)** — Multi-language stop word filtering — extensible, ISO language codes
- **rkyv (v0.8.10)** — Zero-copy statistics serialization — consistent with existing posting list format

### Expected Features

**Must have (table stakes):**
- Standard BM25 formula with configurable k1 (default 1.2) and b (default 0.75) parameters — expected by anyone using BM25
- Document frequency (DF) and term frequency (TF) statistics collection — required for scoring
- Document length tracking per document — required for length normalization
- Average document length (avgdl) per segment — required for BM25 formula
- Multi-term query scoring with score aggregation — users query multiple terms

**Should have (competitive):**
- Hybrid search integration (BM25 + vector similarity) — core differentiator for vector databases
- Score normalization and fusion (linear combination or RRF) — makes hybrid results meaningful
- Top-K retrieval with combined ranking — expected search result format
- Per-segment statistics with cross-segment aggregation — enables distributed architecture

**Defer (v2+):**
- BM25F (field-weighted variant) — requires significant schema changes
- BM25+ (lower-bounded variant) — nice optimization, not essential
- Phrase queries with position scoring — increases index size 3-5x
- Complex query language (AND/OR/NOT) — out of scope per PROJECT.md

### Architecture Approach

BM25 integrates into MuopDB's segment architecture by storing per-segment statistics files alongside existing term indices. The three-tier statistics model (collection, term, document) mirrors Lucene's proven patterns while adapting to MuopDB's multi-user DashMap isolation and mmap'd segment storage.

**Major components:**
1. **Bm25SegmentStats** — Stores per-segment collection stats (doc_count, avgdl) and term stats (doc_freq map)
2. **BM25Scorer** — Query-time scoring component computing IDF and TF components per document
3. **DocumentNorms** — Per-document field lengths for length normalization, stored as encoded 1-byte norms

### Critical Pitfalls

1. **Negative IDF values** — Terms in >50% of documents get negative IDF, penalizing matches. Mitigation: floor IDF at 0 in formula.

2. **Segment-level statistics skew** — BM25 scores computed per-segment use local statistics, making scores incomparable across segments. Mitigation: store global N and avgdl, aggregate at query time.

3. **Hybrid search score mismatch** — BM25 scores (unbounded 0-20+) vs vector similarity (0-1 range) creates imbalance. Mitigation: normalize both to [0,1] before fusion with configurable weights.

4. **Empty collection division by zero** — avgdl calculation with doc_count=0 causes panic. Mitigation: defensive check returning neutral value (1.0).

5. **Statistics staleness during segment merge** — Merged segments may have stale statistics. Mitigation: recompute all statistics post-merge, version statistics for cache invalidation.

## Implications for Roadmap

### Phase 1: Core BM25 Infrastructure
**Rationale:** Foundation layer — statistics structures and scoring formula must exist before index integration.
**Delivers:** Bm25Config, CollectionStatistics, TermStatistics, DocumentNorms structs with serialization
**Addresses:** Table stakes features (configurable k1/b, statistics collection)
**Avoids:** Pitfall 4 (empty collection), Pitfall 5 (parameter validation), Pitfall 3 (FP precision loss)

### Phase 2: Index Building Integration
**Rationale:** Modify existing TermBuilder/MultiTermBuilder to collect statistics during document indexing.
**Delivers:** Modified term builders with BM25 stats collection, BM25StatsWriter for serialization
**Uses:** unicode-segmentation, rust-stemmers, stop-words for tokenization
**Implements:** StatsCollector component from ARCHITECTURE.md
**Avoids:** Pitfall 7 (missing doc length tracking)

### Phase 3: Query Scoring (Read Path)
**Rationale:** Enable BM25 scoring for text-only queries before hybrid integration.
**Delivers:** BM25StatsReader, IDF computation, BM25Scorer, MultiTermIndex extension
**Implements:** Scorer component, pre-computed IDF pattern
**Avoids:** Pitfall 1 (negative IDF), Pitfall 2 (segment skew via aggregation)

### Phase 4: Hybrid Search Integration
**Rationale:** Combine BM25 with existing vector search for unified ranking.
**Delivers:** Score normalization, linear fusion with alpha parameter, top-K aggregation
**Implements:** Hybrid search patterns from FEATURES.md
**Avoids:** Pitfall 6 (score mismatch)

### Phase Ordering Rationale

- **Statistics before scoring:** Must collect document lengths and term frequencies during indexing before query-time scoring works
- **Text-only before hybrid:** Validate BM25 implementation independently before adding fusion complexity
- **Per-segment first, aggregation second:** Each segment must produce valid local scores before cross-segment aggregation
- **Index build path before read path:** Write path modifications (TermBuilder) must exist before read path (Scorer) can function

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 4 (Hybrid Search):** Score normalization algorithms (min-max vs z-score vs rank-based) need experimentation; fusion strategy (linear vs RRF) impacts API design
- **Phase 3 (Query Scoring):** Multi-segment aggregation strategy needs benchmarking for latency targets

Phases with standard patterns (skip research-phase):
- **Phase 1:** Well-documented structures (Lucene CollectionStatistics pattern)
- **Phase 2:** Standard tokenization pipeline (unicode-segmentation docs are comprehensive)

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | unicode-segmentation is industry standard; rkyv already proven in codebase |
| Features | HIGH | Based on Lucene/Elasticsearch/Tantivy implementations — well-established domain |
| Architecture | HIGH | Lucene's three-tier statistics model is battle-tested; MuopDB segment integration is straightforward |
| Pitfalls | HIGH | Derived from official Lucene source, Elasticsearch docs, Stanford IR textbook |

**Overall confidence:** HIGH

### Gaps to Address

- **Performance characteristics:** Need benchmarks for TF/IDF lookup overhead during query — addressed via criterion benchmarks in Phase 3
- **Score normalization strategy:** Research identified multiple approaches but no definitive "best" — resolve via experimentation during Phase 4 implementation
- **Multi-language tokenization:** Default English focus; extension to other languages deferred but documented via stop-words ISO support

## Sources

### Primary (HIGH confidence)
- Lucene BM25Similarity.java (v9.0.0) — Official implementation, formula verification
- Elasticsearch Similarity Settings — Production BM25 configuration patterns
- Tantivy v0.26.0 docs — Fieldnorm compression, Rust-specific patterns
- unicode-segmentation v1.13.2 docs — Unicode Annex #29 compliance

### Secondary (MEDIUM confidence)
- bm25 crate v2.3.2 — Reference implementation for formula verification
- Weaviate Hybrid Search docs — Fusion strategies (RRF, relative score)
- Wikipedia Okapi BM25 — Formula derivation and variants
- Stanford IR Textbook — Theoretical foundation for IDF negativity

### Tertiary (LOW confidence)
- Elasticsearch Practical BM25 Blog — Shard scoring insights (vendor content)
- Xapian docs — BM25F variant patterns (less widely used)

---
*Research completed: 2025-04-05*
*Ready for roadmap: yes*
