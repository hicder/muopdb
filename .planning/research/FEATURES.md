# Feature Landscape: BM25 Ranking for MuopDB

**Domain:** Text search ranking algorithm for hybrid vector+text retrieval
**Researched:** 2026-04-05
**Confidence:** HIGH

## Table Stakes

Features users expect from any BM25 implementation. Missing these = product feels incomplete.

### Core Scoring Algorithm

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **Standard BM25 Formula** | Industry standard since 1994; used by Elasticsearch, Lucene, Tantivy | Low | `score(D,Q) = Σ IDF(qi) * (f(qi,D) * (k1 + 1)) / (f(qi,D) + k1 * (1 - b + b * |D|/avgdl))` |
| **Configurable k1 Parameter** | Controls term frequency saturation; standard range 1.2-2.0 | Low | Default: 1.2. Higher values = more term frequency influence before saturation |
| **Configurable b Parameter** | Controls length normalization; standard range 0-1 | Low | Default: 0.75. b=0 = no length normalization; b=1 = full normalization |
| **IDF Calculation** | Required for term weighting based on rarity | Low | Standard formula: `ln((N - n(qi) + 0.5) / (n(qi) + 0.5) + 1)` |

### Index Statistics (Required at Build Time)

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **Document Frequency (DF) per Term** | Required for IDF calculation | Low | Store `n(qi)` = number of documents containing term qi |
| **Term Frequency per Document** | Required for TF component | Low | MuopDB already has posting lists; need term counts per doc |
| **Document Length per Document** | Required for length normalization | Medium | Store `|D|` = number of terms (not bytes) in each document |
| **Average Document Length (avgdl)** | Required for length normalization | Low | Store `avgdl` = total terms / total documents per segment |
| **Total Document Count (N)** | Required for IDF calculation | Low | Total documents in the collection/segment |

### Query-Time Features

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **Multi-Term Query Scoring** | Users expect to query multiple terms and get ranked results | Medium | Sum scores for all query terms; requires posting list merging |
| **Per-Document Score Computation** | Users need relevance scores | Low | Calculate BM25 score for query-document pair |
| **Top-K Retrieval** | Return most relevant documents only | Medium | Requires efficient pruning during posting list traversal |

## Differentiators

Features that set a BM25 implementation apart from basic offerings.

### Advanced Scoring Variants

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **BM25F (Field-Weighted)** | Different weights for different fields (title vs body) | High | Requires per-field statistics and weighted stream combination. Weaviate, Xapian implement this. |
| **BM25+ (Lower-bounded)** | Prevents unfair scoring of long documents that match query | Medium | Adds δ parameter (default 1.0) to ensure minimum score contribution. Addresses known BM25 deficiency. |
| **BM11/BM15** | Extreme parameter variants | Low | b=1 (BM11) = full length normalization; b=0 (BM15) = no normalization. Edge cases of standard BM25. |

### Storage & Performance Optimizations

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Compressed Document Lengths** | Reduces memory/storage for doc length stats | Medium | Tantivy uses 1-byte fieldnorm_id with logarithmic compression (stores ~|D|/avgdl efficiently) |
| **Per-Segment Statistics** | Enables distributed/segmented index architecture | Medium | Elasticsearch, Tantivy compute stats per segment; global stats at query time. Aligns with MuopDB's segment-based architecture |
| **Lazy Statistics Loading** | Minimize I/O for unused terms | Low | Load DF/IDF stats only when terms are queried |
| **Cached Score Computation** | Speed up repeated queries | Medium | Cache IDF values and per-term score components |

### Hybrid Search Integration

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Score Fusion (RRF)** | Combine BM25 + vector scores effectively | Medium | Reciprocal Rank Fusion: `score = Σ 1/(k + rank)` where k=60 typically. Used by Weaviate. |
| **Weighted Linear Combination** | Simple alpha blending of scores | Low | `final_score = alpha * bm25_score + (1-alpha) * vector_score`. alpha=0.75 is common default. |
| **Cross-Encoders for Re-ranking** | Use neural model to re-rank BM25 candidates | High | Retrieve top-K with BM25, then re-rank with cross-encoder. Improves quality at cost of latency. |
| **Query-Time Score Normalization** | Make BM25 and vector scores comparable | Medium | Normalize both score ranges to [0,1] before fusion. Different approaches: min-max, z-score, softmax. |

### API & Usability Features

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Explain Scoring** | Debug why documents rank as they do | Medium | Return breakdown: IDF component, TF component, length normalization per term. Elasticsearch, Lucene provide this. |
| **Query Boosting** | Allow users to emphasize specific terms | Low | Multiply term score by boost factor. Tantivy `BoostQuery` wrapper pattern. |
| **Phrase Queries** | Match exact phrase sequences | Medium | Requires position information in index (not just posting lists). Increases index size significantly. |
| **Query-Time k1/b Override** | Dataset-specific tuning without rebuild | Low | Allow per-query parameter override for experimentation |

## Anti-Features

Features to explicitly NOT build.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| **Full TF-IDF Support** | BM25 supersedes TF-IDF; no need for legacy | Implement BM25 only; it's the modern standard |
| **Real-Time Index Updates** | Significant complexity for mutable BM25 indices | Follow MuopDB's existing pattern: immutable segments with periodic merges. Rebuild stats on merge. |
| **Complex Query Language (AND/OR/NOT)** | Requires full query parser; out of scope per PROJECT.md | Start with simple bag-of-words; boolean queries can be added later if needed |
| **Position/Proximity Scoring** | Increases index size 3-5x; complex to implement | Defer to future phase; basic BM25 without proximity is table stakes |
| **Per-Field BM25 Parameters** | Adds complexity; rarely needed in practice | Use global k1/b parameters; add field boosting via BM25F if needed later |
| **Automatic Parameter Tuning** | Requires training data and search relevance judgments | Document recommended k1/b ranges; let users tune manually |

## Feature Dependencies

```
BM25 Scoring
├── Document Length Statistics (required)
│   ├── Per-document term count
│   └── Average document length (avgdl)
├── Term Frequency per Document (required)
│   └── Already available in posting lists
├── Document Frequency per Term (required)
│   └── Count of docs containing each term
└── IDF Precomputation (optional optimization)
    └── Cache IDF values at query time

Hybrid Search
├── BM25 Scoring (required)
├── Vector Similarity (required - already exists)
├── Score Normalization (required)
└── Fusion Algorithm (required)
    ├── RRF (ranked-based)
    └── Linear combination (score-based)

BM25F (Field-Weighted)
├── Per-field document lengths (required)
├── Per-field term frequencies (required)
├── Per-field avgdl (required)
└── Field weights configuration (required)
```

## MVP Recommendation

### Phase 1: Core BM25 (Minimum Viable)

Priority order for initial implementation:

1. **Document length collection during index build**
   - Add term count per document to TermBuilder
   - Store doc lengths alongside posting lists

2. **Statistics storage per segment**
   - avgdl per segment
   - Total document count (N)
   - Document frequency (DF) per term

3. **Basic BM25 scoring with configurable k1/b**
   - Standard formula implementation
   - Default k1=1.2, b=0.75

4. **Multi-term query support**
   - Sum scores across query terms
   - Basic posting list intersection/union

### Phase 2: Hybrid Search Integration

5. **Score normalization** - Make BM25 and vector scores comparable
6. **Linear fusion with alpha parameter** - Simple weighted combination
7. **Top-K retrieval with hybrid ranking** - Return fused results

### Defer to Future Phases

- BM25F (field-weighted variant) - Requires significant schema changes
- BM25+ (lower-bounded) - Nice-to-have optimization
- Phrase queries - Requires position index
- RRF fusion - More complex than linear; can add later
- Explain API - Debugging feature

## Major Search Engine Implementations

### Elasticsearch / Lucene (Java)

| Feature | Implementation |
|---------|---------------|
| **Default scoring** | BM25 since v5.0 (was TF-IDF before) |
| **Parameters** | k1 (default 1.2), b (default 0.75), discountOverlaps (default true) |
| **Statistics** | Per-segment with global collection stats |
| **IDF formula** | `log(1 + (docCount - docFreq + 0.5)/(docFreq + 0.5))` |
| **Length norm** | Encoded in single byte with compression |
| **Extensions** | Scripted similarity for custom scoring |
| **Hybrid search** | `semantic_text` field type combining BM25 + dense vectors |

**Key insight from Lucene:** Uses `discountOverlaps` parameter to handle position increment gaps (e.g., synonyms shouldn't count toward document length).

### Tantivy (Rust)

| Feature | Implementation |
|---------|---------------|
| **Default scoring** | BM25 |
| **Fieldnorm compression** | 1-byte `fieldnorm_id` with logarithmic encoding |
| **Fieldnorm encoding** | `id_to_fieldnorm(id)` = roughly linear for small values, compresses large values |
| **API** | `Bm25StatisticsProvider` trait for stats access |
| **Boosting** | `BoostQuery` wrapper for query-time score multiplication |
| **Multi-term** | Query parser with automatic boolean query construction |

**Key insight from Tantivy:** Document lengths are approximated with 1-byte encoding. This is acceptable because BM25 uses length in a ratio (`|D|/avgdl`), so precise values aren't critical.

### Weaviate (Go)

| Feature | Implementation |
|---------|---------------|
| **BM25 variant** | Standard BM25 + BM25F |
| **Hybrid approach** | Parallel BM25 + vector search, then fusion |
| **Fusion algorithms** | `rankedFusion` (RRF), `relativeScoreFusion` |
| **Alpha parameter** | 0 = pure BM25, 1 = pure vector, 0.75 = default |
| **Performance** | Async parallel execution of both search types |

**Key insight from Weaviate:** Two fusion strategies - RRF (rank-based, more stable) and relative score fusion (score-based, more sensitive to score distributions).

### Xapian (C++)

| Feature | Implementation |
|---------|---------------|
| **Variants** | BM25, BM25F (with field weights) |
| **Parameters** | k1, k2 (for query term weighting), b, min_normlen |
| **k2 parameter** | Extra parameter for query term frequency weighting (rarely used) |

## Implementation Complexity Summary

| Feature Category | Complexity | Effort Estimate | Risk Level |
|-----------------|------------|-----------------|------------|
| Core BM25 scoring | Low | 1-2 weeks | Low |
| Statistics collection | Medium | 2-3 weeks | Medium |
| Multi-term queries | Medium | 2-3 weeks | Medium |
| Hybrid fusion | Medium | 2-3 weeks | Medium |
| BM25F (field-weighted) | High | 4-6 weeks | High |
| Score explanation | Medium | 1-2 weeks | Low |
| Performance optimization | Medium | 2-3 weeks | Low |

## Recommended Defaults

Based on industry standard implementations:

```rust
pub struct Bm25Config {
    /// Term frequency saturation parameter
    /// Range: [0, ∞), typically [1.2, 2.0]
    /// Higher = more influence from term frequency before saturation
    pub k1: f32, // Default: 1.2
    
    /// Length normalization parameter
    /// Range: [0, 1]
    /// 0 = no length normalization
    /// 1 = full length normalization
    pub b: f32, // Default: 0.75
    
    /// Whether to discount overlapping tokens from document length
    /// (e.g., synonyms with position increment 0)
    pub discount_overlaps: bool, // Default: true
}

impl Default for Bm25Config {
    fn default() -> Self {
        Self {
            k1: 1.2,
            b: 0.75,
            discount_overlaps: true,
        }
    }
}
```

## Sources

- **Elasticsearch BM25 Similarity**: https://www.elastic.co/docs/reference/elasticsearch/index-settings/similarity (HIGH confidence - official docs)
- **Lucene BM25Similarity JavaDoc**: https://lucene.apache.org/core/9_0_0/core/org/apache/lucene/search/similarities/BM25Similarity.html (HIGH confidence - official API)
- **Tantivy FieldNorm Reader**: https://github.com/quickwit-oss/tantivy/blob/main/src/fieldnorm/reader.rs (HIGH confidence - source code)
- **Tantivy Documentation**: https://docs.rs/tantivy/latest/tantivy/ (HIGH confidence - official docs)
- **Wikipedia Okapi BM25**: https://en.wikipedia.org/wiki/Okapi_BM25 (MEDIUM confidence - community edited but well-established facts)
- **Weaviate Hybrid Search**: https://weaviate.io/blog/hybrid-search-explained (MEDIUM confidence - vendor blog but technically accurate)
- **Elasticsearch Semantic Text**: https://www.elastic.co/docs/reference/elasticsearch/mapping-reference/semantic-text (HIGH confidence - official docs)
- **Wikipedia TF-IDF**: https://en.wikipedia.org/wiki/Tf%E2%80%93idf (MEDIUM confidence - historical context)
