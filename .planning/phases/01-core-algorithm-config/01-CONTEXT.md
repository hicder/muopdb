# Phase 01: Core Algorithm & Config - Context

**Gathered:** 2026-04-05
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement the core BM25 scoring formula with configurable k1 and b parameters. This phase delivers the foundational scoring infrastructure that subsequent phases will build upon for statistics collection and query scoring.

</domain>

<decisions>
## Implementation Decisions

### Parameter Defaults
- **D-01:** Use Lucene/Elasticsearch defaults: k1 = 1.2, b = 0.75
- **D-02:** Parameter ranges: k1 ∈ (0, ∞), b ∈ [0, 1] with validation on construction

### IDF Edge Handling
- **D-03:** Floor IDF at 0 to prevent negative scores for high-frequency terms (terms appearing in >50% of documents)
- **D-04:** Use standard BM25 IDF formula: log(1 + (N - n + 0.5) / (n + 0.5)) where N = total docs, n = docs containing term

### TF Precision
- **D-05:** Use f64 for all intermediate BM25 calculations to maintain precision
- **D-06:** Return f32 scores for API consistency with existing vector similarity scores

### Statistics Storage
- **D-07:** Store per-segment statistics in dedicated bm25_stats/ subdirectory alongside segment files
- **D-08:** Use rkyv serialization for zero-copy deserialization at query time (consistent with existing posting list format)

### the agent's Discretion
- Tokenization pipeline design (will adapt from existing TermIndex patterns)
- Exact struct layout for BM25Config (follow existing Config patterns in codebase)
- Unit test organization (follow existing test patterns)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Existing Infrastructure
- `.planning/codebase/ARCHITECTURE.md` — MuopDB segment architecture overview
- `.planning/codebase/STACK.md` — Technology stack including rkyv serialization
- `.planning/codebase/CONVENTIONS.md` — Code conventions for the codebase
- `rs/index/src/multi_terms/index.rs` — Existing TermIndex and MultiTermIndex patterns
- `rs/index/src/terms/index.rs` — TermIndex implementation

### BM25 Formula References
- Lucene BM25Similarity.java (v9.0.0) — Reference implementation
- Wikipedia Okapi BM25 — Formula specification

### Configuration Patterns
- `rs/config/src/collection.rs` — CollectionConfig pattern to follow
- `rs/quantization/src/pq/mod.rs` — ProductQuantizerConfig pattern

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **Config structs**: Follow patterns in `rs/config/src/collection.rs` and `rs/quantization/src/pq/mod.rs` for BM25Config
- **Serialization**: Use existing rkyv patterns from posting list storage
- **Validation**: Use standard Rust validation patterns (Result<(), ValidationError>)

### Established Patterns
- **Builder pattern**: Used extensively in SpannBuilder, IvfBuilder — BM25 stats collection should follow similar pattern
- **Segment-based storage**: Statistics stored per-segment, aggregated at query time
- **Multi-user isolation**: Per-user statistics via DashMap (follow MultiTermIndex pattern)

### Integration Points
- **TermIndex**: BM25 stats will extend existing term index structures
- **MultiTermIndex**: BM25Scorer will integrate with existing lazy-loading pattern
- **Segment lifecycle**: Stats written during segment flush, read during segment search

</code_context>

<specifics>
## Specific Ideas

No specific requirements — standard BM25 implementation following Lucene/Elasticsearch patterns.

Key formula components to implement:
1. IDF = ln(1 + (N - n + 0.5) / (n + 0.5))
2. TF_component = (f * (k1 + 1)) / (f + k1 * (1 - b + b * (dl / avgdl)))
3. Score = IDF * TF_component

Where:
- N = total document count in segment
- n = document frequency (number of docs containing term)
- f = term frequency in document
- dl = document length (number of terms in document)
- avgdl = average document length in segment

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 01-core-algorithm-config*
*Context gathered: 2026-04-05*
