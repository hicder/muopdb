# Technology Stack: BM25 Ranking Algorithm

**Project:** MuopDB BM25 Support  
**Researched:** 2025-04-05  
**Domain:** Text ranking for vector database hybrid search

## Executive Summary

For MuopDB's BM25 implementation, we recommend a **custom-built BM25 scoring layer** integrated with the existing `multi_terms` infrastructure rather than adopting a full external search engine. This approach aligns with MuopDB's segment-based architecture and avoids the architectural impedance mismatch that would come with integrating Tantivy or other full-text engines.

The 2025 Rust ecosystem provides excellent primitives for tokenization (`unicode-segmentation`, `rust-stemmers`) and math utilities, but BM25 scoring itself is straightforward enough to implement directly while maintaining control over the storage format and query execution.

## Recommended Stack

### Core Algorithm Libraries

| Technology | Version | Purpose | Confidence |
|------------|---------|---------|------------|
| **Custom Implementation** | N/A | BM25 scoring formula, IDF calculation, term frequency tracking | HIGH |
| `unicode-segmentation` | `1.13` | Unicode-compliant word tokenization | HIGH |
| `rust-stemmers` | `1.2` (existing) | Snowball stemming algorithms | HIGH |
| `stop-words` | `0.10` | Multi-language stop word lists | MEDIUM |

**Rationale for custom implementation:**

BM25 is a well-defined formula with no ambiguous implementation details:

```
score(D, Q) = Σ IDF(qi) * (f(qi, D) * (k1 + 1)) / (f(qi, D) + k1 * (1 - b + b * |D| / avgdl))
IDF(qi) = ln((N - n(qi) + 0.5) / (n(qi) + 0.5) + 1)
```

External crates like `bm25` (v2.3.2) provide excellent standalone implementations, but MuopDB requires:
1. Integration with existing Elias-Fano encoded posting lists
2. Per-segment statistic storage (aligned with segment architecture)
3. Multi-user isolation via `DashMap<u128, ...>`
4. Hybrid scoring with vector similarity

The `bm25` crate assumes in-memory document storage and produces sparse embeddings—useful for reference, but not a drop-in fit for MuopDB's mmap'd segment architecture.

### Tokenization and Text Processing Stack

| Technology | Version | Purpose | Confidence |
|------------|---------|---------|------------|
| `unicode-segmentation` | `1.13.2` | Word boundary detection per Unicode Annex #29 | HIGH |
| `rust-stemmers` | `1.2.0` | Stemming (English default, 17 languages available) | HIGH |
| `stop-words` | `0.10.0` | Stop word filtering (130+ languages) | MEDIUM |

**Tokenization Pipeline:**

```rust
// Recommended processing order
1. Unicode word segmentation (unicode-segmentation)
2. Lowercasing
3. Stop word removal (stop-words)
4. Stemming (rust-stemmers)
5. Token ID assignment via OnDiskOrderedMap
```

**Why not `regex` for tokenization?**  
Unicode word boundaries are complex (handling emojis, compound characters, non-Latin scripts). `unicode-segmentation` provides standard-compliant handling with zero-allocation iterators and `#![no_std]` support.

**Why not Tantivy's tokenizer API?**  
Tantivy's `tantivy-tokenizer-api` (v0.7) is well-designed but brings in the full Tantivy ecosystem. For MuopDB's narrower use case, direct use of underlying primitives provides more control and fewer dependencies.

### Statistics Storage and Serialization

| Technology | Version | Purpose | Confidence |
|------------|---------|---------|------------|
| `rkyv` | `0.8.10` (existing) | Zero-copy deserialization of term statistics | HIGH |
| `serde` + `serde_json` | `1.0` (existing) | Configuration, debugging output | HIGH |
| Custom segment metadata | N/A | Per-segment DF/TF statistics | HIGH |

**Storage Format Recommendation:**

```rust
// Per-segment BM25 statistics (stored alongside segment)
#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct Bm25SegmentStats {
    /// Total documents in segment
    pub doc_count: u32,
    /// Average document length (in tokens)
    pub avgdl: f32,
    /// Document frequency per term (term_id -> df)
    pub doc_frequencies: HashMap<u64, u32>,
    /// Per-document term frequencies (doc_id -> term_freqs)
    pub term_freqs: Vec<HashMap<u64, u32>>,
    /// Per-document token counts (for length normalization)
    pub doc_lengths: Vec<u32>,
}
```

**Why `rkyv`:** MuopDB already uses `rkyv` for zero-copy deserialization of posting lists. Extending this to BM25 statistics maintains consistency and performance characteristics.

### Testing and Benchmarking

| Technology | Version | Purpose | Confidence |
|------------|---------|---------|------------|
| `criterion` | `0.5` | Micro-benchmarks for scoring functions | HIGH |
| `proptest` | `1.0+` | Property-based testing of IDF/TF calculations | MEDIUM |
| Custom test corpus | N/A | Validation against reference BM25 implementations | HIGH |

**Recommended Testing Approach:**

1. **Unit tests:** Verify IDF formula against known values (N=1000, df=10 → expected IDF)
2. **Property tests:** Invariants (score ≥ 0, score increases with TF up to saturation)
3. **Reference tests:** Compare scores against `bm25` crate output for same corpus
4. **Integration tests:** End-to-end with sample Wikipedia articles

## Alternatives Considered

### Tantivy (v0.26.0)

**Why NOT to use:**

| Concern | Detail |
|---------|--------|
| Architecture mismatch | Tantivy owns the index structure; MuopDB has segment-based architecture |
| Storage format | Tantivy uses custom Directory abstraction; MuopDB uses mmap + async block I/O |
| Multi-user model | Tantivy assumes single-tenant indices; MuopDB uses DashMap per-user isolation |
| Feature weight | Tantivy is 14.8k stars, full search engine—massive overkill for BM25 scoring only |

**When WOULD you use Tantivy:**
- Building a new search engine from scratch (not adding to existing vector DB)
- Need full query language (AND/OR/NOT), phrase queries, faceted search
- Can adopt Tantivy's segment/merging architecture wholesale

### `bm25` Crate (v2.3.2)

**Why NOT to use as primary:**

| Concern | Detail |
|---------|--------|
| In-memory only | Designed for sparse embeddings in vector DBs, not on-disk segments |
| Document model | Assumes documents are strings; MuopDB has point IDs with term mappings |
| Custom tokenizer trait | Good design, but requires adapter layer to MuopDB's existing term indexing |

**Why TO reference it:**
- Excellent reference implementation of BM25 formula
- Well-tested tokenization pipeline with language detection
- `parallelism` feature with Rayon for batch processing patterns

### LanceDB (v0.27.2)

LanceDB uses `tokenizers` crate (HuggingFace) for embeddings and DataFusion for full-text search. Not directly applicable to MuopDB's architecture but shows industry trend toward:
- Hybrid vector + text search
- Arrow-native storage

## Installation

```bash
# Add to workspace dependencies
cargo add --package index unicode-segmentation@1.13.2
cargo add --package index stop-words@0.10.0

# rust-stemmers already present at 1.2.0
```

**Feature flags to enable:**

```toml
[dependencies]
unicode-segmentation = { version = "1.13", default-features = false }
stop-words = { version = "0.10", default-features = false, features = ["iso"] }
```

## Integration Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    BM25 Integration Layer                    │
├─────────────────────────────────────────────────────────────┤
│  Tokenizer (unicode-segmentation → stem → stop-word filter) │
│                        ↓                                    │
│  TermIndex (existing) → term ID → Posting List (Elias-Fano) │
│                        ↓                                    │
│  Bm25Scorer: IDF lookup + TF calculation + Length norm      │
│                        ↓                                    │
│  Score combination with vector similarity (hybrid search)   │
└─────────────────────────────────────────────────────────────┘
```

**Per-Segment Stats Storage:**

```
segment/
├── terms/
│   ├── term_index          # Existing
│   └── posting_lists       # Existing (Elias-Fano)
└── bm25_stats.rkyv         # NEW: Bm25SegmentStats
```

## Confidence Assessment

| Area | Confidence | Reason |
|------|------------|--------|
| Core BM25 formula | HIGH | Well-defined standard, straightforward math |
| Tokenization stack | HIGH | `unicode-segmentation` is industry standard |
| Storage approach | HIGH | Extends existing `rkyv` + segment pattern |
| Integration complexity | MEDIUM | Requires careful design for per-user statistics |
| Performance characteristics | MEDIUM | Need benchmarks for TF/IDF lookup overhead |

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| TF calculation overhead during query | Pre-compute per-doc term frequencies during indexing |
| avgdl drift with document addition | Recompute on segment merge; accept small drift for mutable segments |
| Tokenization performance | Benchmark against regex approach; optimize hot paths |
| Multi-language support | Start with English only; `stop-words` + `rust-stemmers` extensible |

## Sources

- Tantivy v0.26.0 docs: https://docs.rs/tantivy/0.26.0/tantivy/ (verified: BM25 same as Lucene)
- `bm25` crate v2.3.2: https://docs.rs/bm25/2.3.2/bm25/ (verified: latest features)
- `unicode-segmentation` v1.13.2: https://docs.rs/unicode-segmentation/1.13.2/unicode_segmentation/
- `rust-stemmers` v1.2.0: https://docs.rs/rust-stemmers/1.2.0/rust_stemmers/
- `stop-words` v0.10.0: https://docs.rs/stop-words/0.10.0/stop_words/
- MuopDB existing codebase: `rs/index/src/multi_terms/`, `rs/index/src/terms/`

---

**Recommendation:** Proceed with custom BM25 implementation using `unicode-segmentation`, `rust-stemmers` (existing), and `stop-words`. Reference `bm25` crate implementation for formula verification. Avoid Tantivy due to architectural mismatch.
