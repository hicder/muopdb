# BM25 Architecture Patterns

**Domain:** BM25 Text Ranking Algorithm
**Researched:** 2026-04-05
**Confidence:** HIGH (based on Lucene/Elasticsearch implementation patterns and MuopDB codebase analysis)

## Executive Summary

BM25 systems follow a three-tier statistics architecture: **collection-level statistics** (document count, average length), **term-level statistics** (document frequency, total term frequency), and **document-level statistics** (field length/norms). The query pipeline separates pre-computation of collection-level weights (IDF) from per-document scoring. Integration with MuopDB's segment-based architecture requires storing per-segment statistics alongside existing Elias-Fano encoded posting lists.

## BM25 Component Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           BM25 System Architecture                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────┐    ┌─────────────────────┐    ┌───────────────────┐ │
│  │   Stats Collector   │    │    Index Writer     │    │      Scorer       │ │
│  │                     │    │                     │    │                   │ │
│  │ • Document counter  │───▶│ • Term dictionary   │───▶│ • IDF cache       │ │
│  │ • Term frequencies  │    │ • Posting lists     │    │ • TF computation  │ │
│  │ • Doc lengths       │    │ • Statistics store  │    │ • Length norm     │ │
│  │ • Per-user stats    │    │ • Segment metadata  │    │ • Score aggregation│ │
│  └─────────────────────┘    └─────────────────────┘    └───────────────────┘ │
│           │                          │                         │            │
│           ▼                          ▼                         ▼            │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                     Statistics Storage Layer                           │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │ │
│  │  │ Collection   │  │   Term       │  │  Document    │  │  Config    │ │ │
│  │  │ Stats        │  │   Stats      │  │  Norms       │  │  (k1, b)   │ │ │
│  │  │ • doc_count  │  │ • doc_freq   │  │ • length     │  │            │ │ │
│  │  │ • avgdl      │  │ • total_tf   │  │ • norm_enc   │  │            │ │ │
│  │  │ • sumTotalTF │  │              │  │              │  │            │ │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  └────────────┘ │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Component Boundaries

| Component | Responsibility | Input | Output | Communicates With |
|-----------|---------------|-------|--------|-------------------|
| **StatsCollector** | Aggregates statistics during index build | Documents, terms, token streams | CollectionStats, TermStats, DocNorms | IndexWriter, Scorer (at query time) |
| **IndexWriter** | Persists inverted index + statistics | Term postings, statistics structs | Segment files on disk | StatsCollector, TermIndex |
| **Scorer** | Computes BM25 scores at query time | Query terms, segment stats | Ranked document list | Stats store, Posting list iterators |
| **StatsStore** | Manages per-segment statistics persistence | Statistics structs | Serialized stats files | All components |

### Data Structures (Lucene-inspired)

```rust
/// Collection-level statistics (per segment/user)
pub struct CollectionStatistics {
    pub field: String,           // Field name (e.g., "content")
    pub max_doc: u64,            // Total documents in segment
    pub doc_count: u64,          // Documents with this field
    pub sum_total_term_freq: u64, // Total tokens (sum of all doc lengths)
    pub sum_doc_freq: u64,       // Sum of unique terms per doc
}

impl CollectionStatistics {
    /// Average document length = sum_total_term_freq / doc_count
    pub fn avg_field_length(&self) -> f32 {
        self.sum_total_term_freq as f32 / self.doc_count as f32
    }
}

/// Term-level statistics (per segment/user)
pub struct TermStatistics {
    pub term: String,            // Term text
    pub doc_freq: u64,           // Documents containing this term
    pub total_term_freq: u64,  // Total occurrences across all docs
}

/// Document-level norm (length factor)
pub struct DocumentNorm {
    pub doc_id: u128,            // Document identifier
    pub field_length: u32,       // Number of tokens in field
    pub encoded_norm: u8,        // Lucene-style encoded length (for fast scoring)
}

/// BM25 parameters
pub struct BM25Params {
    pub k1: f32,                 // Term frequency saturation (default: 1.2)
    pub b: f32,                  // Length normalization (default: 0.75)
}
```

## Integration with MuopDB Segment Architecture

### Segment Extension for BM25

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    MuopDB Segment with BM25 Support                          │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                        ImmutableSegment<Q>                               │  │
│  │                                                                        │  │
│  │  ┌──────────────────────┐    ┌──────────────────────┐                   │  │
│  │  │   MultiSpannIndex    │    │  MultiTermIndex      │                   │  │
│  │  │   (vector search)    │    │  (text + BM25)       │                   │  │
│  │  │                      │    │                      │                   │  │
│  │  │ • Spann indices      │    │ • TermIndex per user │                   │  │
│  │  │ • HNSW centroids     │    │ • Posting lists      │                   │  │
│  │  │ • IVF posting lists  │    │ • BM25 stats ────────────┐               │  │
│  │  └──────────────────────┘    └──────────────────────┘   │               │  │
│  │                                         │               │               │  │
│  │                                         ▼               │               │  │
│  │                            ┌──────────────────────┐    │               │  │
│  │                            │  BM25StatisticsStore │    │               │  │
│  │                            │                      │◀───┘               │  │
│  │                            │ • Per-user stats     │                    │  │
│  │                            │ • CollectionStats    │                    │  │
│  │                            │ • TermStats index    │                    │  │
│  │                            │ • DocumentNorms      │                    │  │
│  │                            └──────────────────────┘                    │  │
│  │                                                                        │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Per-Segment Storage Layout

```
segment_12345/
├── spann_index/               # Existing SPANN vector index
│   ├── centroids/
│   └── posting_lists/
├── terms/                     # Existing term infrastructure
│   ├── combined               # Term dictionary + posting lists (Elias-Fano)
│   └── user_term_index_info   # Per-user offsets
├── bm25_stats/               # NEW: BM25 statistics
│   ├── collection_stats      # Per-user collection statistics
│   ├── term_stats            # Term frequency statistics (term_id → doc_freq)
│   ├── document_norms        # Per-document field lengths
│   └── bm25_config           # k1, b parameters (or use collection defaults)
└── segment_metadata.json
```

### File Formats

**collection_stats** (per user):
```rust
// Serialized format: bincode or rkyv
pub struct UserCollectionStats {
    pub user_id: u128,
    pub doc_count: u64,           // Number of documents for this user
    pub sum_total_term_freq: u64, // Sum of all document lengths
    pub sum_doc_freq: u64,        // Sum of unique terms per doc
}
```

**term_stats** (per user, term_id → stats):
```rust
// Stored alongside posting list offsets or in separate file
pub struct TermStatsEntry {
    pub term_id: u64,
    pub doc_freq: u64,           // n(qi): documents containing term
    pub total_term_freq: u64,    // Total occurrences of term
}
```

**document_norms** (per user, doc_id → length):
```rust
// Fixed-size array for O(1) lookup by point_id
// Encoded as Lucene-style norm for fast scoring
pub struct DocumentNorms {
    pub user_id: u128,
    // Vec indexed by point_id (u32) within user's segment
    pub norms: Vec<u8>,  // Encoded length: small values = longer docs
}

// Lucene norm encoding (for reference):
// norm = 1 / sqrt(length) encoded as single byte
// Longer documents → smaller norm values → lower scores
```

## Data Flow

### Index Building Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           Index Building Data Flow                                   │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│   1. DOCUMENT INGESTION                                                              │
│   ┌───────────┐    ┌──────────────┐    ┌──────────────────┐    ┌───────────────┐  │
│   │  Raw Doc  │───▶│  Tokenizer   │───▶│  Token Stream    │───▶│ Term + Count  │  │
│   │ (text)    │    │ (analyze)    │    │  (tokens, freq)  │    │ per document  │  │
│   └───────────┘    └──────────────┘    └──────────────────┘    └───────────────┘  │
│                                                                          │         │
│   2. STATISTICS COLLECTION                                                             │
│                                                                          ▼         │
│   ┌─────────────────────────────────────────────────────────────────────────────┐ │
│   │                         StatsCollector                                        │ │
│   │                                                                             │ │
│   │   For each user/document:                                                    │ │
│   │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │ │
│   │   │ doc_count++ │  │ Add to      │  │ Update      │  │ Store doc length│ │ │
│   │   │             │  │ sumTotalTF  │  │ term doc_freq│  │ in norms table  │ │ │
│   │   └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────┘ │ │
│   └─────────────────────────────────────────────────────────────────────────────┘ │
│                                          │                                          │
│   3. INDEX WRITING                                                                   │
│                                          ▼                                          │
│   ┌─────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌────────────┐ │
│   │  TermDict   │───▶│  Posting Lists  │───▶│  Stats Files    │───▶│  Segment   │ │
│   │  Builder    │    │  (Elias-Fano)   │    │  (bincode)      │    │  TOC       │ │
│   └─────────────┘    └─────────────────┘    └─────────────────┘    └────────────┘ │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### Query Scoring Flow

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                            Query Scoring Data Flow                                    │
├──────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                       │
│   1. QUERY PREPARATION                                                                │
│   ┌────────────┐    ┌─────────────┐    ┌─────────────────┐    ┌──────────────────┐ │
│   │ Query Text │───▶│  Tokenize   │───▶│  Get term IDs   │───▶│  Load segment    │ │
│   │            │    │             │    │  from TermIndex │    │  BM25 stats      │ │
│   └────────────┘    └─────────────┘    └─────────────────┘    └──────────────────┘ │
│                                                                          │            │
│   2. SCORE COMPUTATION (per segment)                                                  │
│                                                                          ▼            │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐ │
│   │                            BM25Scorer                                           │ │
│   │                                                                                 │ │
│   │  For each query term:                                                          │ │
│   │  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────────────────────┐ │ │
│   │  │ Compute IDF      │  │ Load posting     │  │ For each doc in posting list:  │ │ │
│   │  │                  │  │ list iterator    │  │                                │ │ │
│   │  │ idf = ln((N -    │  │                  │  │ score += idf * tf_component    │ │ │
│   │  │  n(q) + 0.5) /   │  │ Elias-Fano       │  │                                │ │ │
│   │  │  (n(q) + 0.5) + 1)│  │ decoder          │  │ tf = freq * (k1 + 1) /         │ │ │
│   │  │                  │  │                  │  │      (freq + k1 * (1 - b +     │ │ │
│   │  │ N = collection   │  │ yields (doc_id,  │  │       b * |D|/avgdl))          │ │ │
│   │  │ doc_count        │  │ term_freq)       │  │                                │ │ │
│   │  │ n(q) = term's    │  │                  │  │ |D| from document_norms        │ │ │
│   │  │ doc_freq         │  │                  │  │ avgdl from collection_stats    │ │ │
│   │  └──────────────────┘  └──────────────────┘  └────────────────────────────────┘ │ │
│   └─────────────────────────────────────────────────────────────────────────────────┘ │
│                                          │                                            │
│   3. SEGMENT AGGREGATION                                                               │
│                                          ▼                                            │
│   ┌────────────────┐    ┌──────────────────┐    ┌────────────────┐    ┌────────────┐  │
│   │ Merge scores   │───▶│  Merge-sort by   │───▶│  Top-K results │───▶│ Return to  │  │
│   │ from all       │    │  score desc      │    │  per segment   │    │ client     │  │
│   │ segments       │    │                  │    │                │    │            │  │
│   └────────────────┘    └──────────────────┘    └────────────────┘    └────────────┘  │
│                                                                                        │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

## Build Order and Dependencies

### Recommended Implementation Sequence

```
Phase 1: Foundation (Stats Collection)
│
├── 1.1 BM25Config struct (k1, b parameters)
│   └── Simple struct, no dependencies
│
├── 1.2 CollectionStatistics struct
│   └── Depends on: Document counting during indexing
│
├── 1.3 TermStatistics struct + collection
│   └── Depends on: TermBuilder modifications
│
└── 1.4 DocumentNorms struct
    └── Depends on: Point ID allocation

Phase 2: Index Building Integration
│
├── 2.1 Modify TermBuilder to collect term frequencies
│   └── Depends on: Phase 1.3
│
├── 2.2 Modify MultiTermBuilder to aggregate per-user stats
│   └── Depends on: 2.1, existing MultiTermBuilder
│
├── 2.3 BM25StatsWriter (serializes stats to disk)
│   └── Depends on: Phase 1, MultiTermWriter pattern
│
└── 2.4 Modify MultiTermWriter to write BM25 stats
    └── Depends on: 2.2, 2.3

Phase 3: Query Scoring (Read Path)
│
├── 3.1 BM25StatsReader (loads stats from disk)
│   └── Depends on: 2.3 format
│
├── 3.2 IDF computation module
│   └── Depends on: CollectionStatistics
│
├── 3.3 BM25Scorer (document scoring)
│   └── Depends on: 3.1, 3.2, DocumentNorms
│
└── 3.4 MultiTermIndex integration for BM25
    └── Depends on: 3.3, existing MultiTermIndex

Phase 4: Hybrid Search Integration
│
├── 4.1 Score normalization (BM25 + vector scores)
│   └── Depends on: 3.4, SPANN search results
│
└── 4.2 Top-K aggregation across modalities
    └── Depends on: 4.1
```

### Dependency Graph

```
                    ┌──────────────────────┐
                    │    BM25Config        │
                    └──────────┬───────────┘
                               │
           ┌───────────────────┼───────────────────┐
           │                   │                   │
           ▼                   ▼                   ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ CollectionStats  │  │   TermStats      │  │  DocumentNorms   │
└────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
         │                     │                     │
         │                     ▼                     │
         │            ┌──────────────────┐          │
         │            │  TermBuilder     │          │
         │            │  (modified)      │          │
         │            └────────┬─────────┘          │
         │                     │                    │
         ▼                     ▼                    ▼
┌─────────────────────────────────────────────────────────────┐
│                    MultiTermBuilder (modified)                │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    BM25StatsWriter                            │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    MultiTermWriter (modified)                 │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    BM25StatsReader                            │
└─────────────────────────────┬───────────────────────────────┘
                              │
           ┌──────────────────┼──────────────────┐
           │                  │                  │
           ▼                  ▼                  ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  IDF Computation │  │  BM25Scorer      │  │ MultiTermIndex   │
└────────┬─────────┘  └────────┬─────────┘  │   (BM25 ext)     │
         │                     │           └────────┬─────────┘
         │                     │                    │
         └──────────┬──────────┘                    │
                    │                             │
                    ▼                             ▼
         ┌────────────────────┐          ┌──────────────────┐
         │   BM25 Search      │          │  Hybrid Search   │
         │   (single modality)│          │  (BM25 + Vector) │
         └────────────────────┘          └──────────────────┘
```

## Architecture Patterns

### Pattern 1: Pre-computed IDF with Segment-Level Stats

**What:** Compute IDF at query time using per-segment statistics, then aggregate scores across segments.

**Why:** Allows segment-level parallelism and follows MuopDB's existing search pattern.

**Implementation:**
```rust
pub struct SegmentBM25Scorer {
    collection_stats: CollectionStatistics,
    term_stats: HashMap<u64, TermStatistics>,  // term_id → stats
    document_norms: Vec<u8>,  // indexed by point_id
    params: BM25Params,
}

impl SegmentBM25Scorer {
    fn compute_idf(&self, term_stats: &TermStatistics) -> f32 {
        let n = term_stats.doc_freq as f32;
        let n_docs = self.collection_stats.doc_count as f32;
        
        // IDF = ln((N - n(q) + 0.5) / (n(q) + 0.5) + 1)
        ((n_docs - n + 0.5) / (n + 0.5) + 1.0).ln()
    }
    
    fn score_document(&self, freq: f32, doc_length: u32) -> f32 {
        let avgdl = self.collection_stats.avg_field_length();
        let norm = 1.0 - self.params.b + self.params.b * (doc_length as f32 / avgdl);
        
        // TF component: f(q,D) * (k1 + 1) / (f(q,D) + k1 * norm)
        (freq * (self.params.k1 + 1.0)) / (freq + self.params.k1 * norm)
    }
}
```

### Pattern 2: Lazy Stats Loading

**What:** Load BM25 statistics on-demand when first query accesses a user's index.

**Why:** Minimizes memory overhead for inactive users; follows existing `MultiTermIndex` lazy loading pattern.

**Implementation:**
```rust
pub struct MultiTermIndex {
    term_indexes: DashMap<u128, Arc<TermIndex>>,
    bm25_stats: DashMap<u128, Arc<BM25Statistics>>,  // Lazy loaded
    user_index_info: TermIndexInfoHashTable,
    base_directory: String,
}

impl MultiTermIndex {
    pub async fn get_bm25_stats(&self, user_id: u128) -> Result<Arc<BM25Statistics>> {
        // Try cache first
        if let Some(stats) = self.bm25_stats.get(&user_id) {
            return Ok(stats.clone());
        }
        
        // Load from disk
        let stats_path = format!("{}/bm25_stats/{}_stats", self.base_directory, user_id);
        let stats = BM25Statistics::load(&stats_path).await?;
        let stats_arc = Arc::new(stats);
        
        self.bm25_stats.insert(user_id, stats_arc.clone());
        Ok(stats_arc)
    }
}
```

### Pattern 3: Per-Segment Statistics Aggregation

**What:** Store complete statistics per segment; collection-level stats computed by aggregation during multi-segment search.

**Why:** Supports segment-level isolation and concurrent segment operations (merge, vacuum).

**Formula adaptation for multi-segment:**
```rust
// For cross-segment BM25, we need to compute
// collection-wide stats by aggregating segment stats

pub struct AggregatedCollectionStats {
    pub total_doc_count: u64,
    pub total_sum_term_freq: u64,
}

impl AggregatedCollectionStats {
    pub fn from_segments(segments: &[&CollectionStatistics]) -> Self {
        let mut total_doc_count = 0;
        let mut total_sum_term_freq = 0;
        
        for seg in segments {
            total_doc_count += seg.doc_count;
            total_sum_term_freq += seg.sum_total_term_freq;
        }
        
        Self { total_doc_count, total_sum_term_freq }
    }
    
    pub fn avgdl(&self) -> f32 {
        self.total_sum_term_freq as f32 / self.total_doc_count as f32
    }
}
```

## Anti-Patterns to Avoid

### Anti-Pattern 1: Global Statistics Store

**What:** A single central statistics file for all segments.

**Why bad:** Breaks segment isolation; concurrent merges become complex; hard to vacuum invalidated documents.

**Instead:** Per-segment statistics with cross-segment aggregation at query time.

### Anti-Pattern 2: Computing IDF Per-Document

**What:** Computing IDF inside the inner scoring loop.

**Why bad:** IDF is constant for a given query/segment combination; recomputing wastes cycles.

**Instead:** Pre-compute IDF once per term per segment, then reuse for all documents.

### Anti-Pattern 3: Storing Raw Document Lengths as Floats

**What:** Storing f32 document lengths in norms table.

**Why bad:** 4x memory overhead vs encoded norms; slower cache performance.

**Instead:** Use Lucene-style single-byte norm encoding: `norm = 1/sqrt(length)` encoded in 1 byte.

## Integration with Existing MuopDB Components

### TermBuilder Extension

```rust
pub struct TermBuilder {
    term_map: OnDiskOrderedMapBuilder,
    next_term_id: u64,
    posting_lists: HashMap<u64, Vec<u32>>,
    
    // NEW: BM25 statistics collection
    term_stats: HashMap<u64, TermStatisticsBuilder>,
    doc_lengths: HashMap<u32, u32>,  // point_id → term count in doc
    
    built: bool,
}

struct TermStatisticsBuilder {
    doc_freq: u64,        // Number of docs containing term
    total_term_freq: u64, // Total occurrences across all docs
}

impl TermBuilder {
    pub fn add(&mut self, point_id: u32, key: String) -> Result<()> {
        // Existing code...
        
        // NEW: Track term statistics
        let term_id = self.persist_and_get_term_id(key);
        let stats = self.term_stats.entry(term_id).or_default();
        stats.total_term_freq += 1;
        
        // Track doc frequency (unique docs per term)
        let doc_list = self.posting_lists.entry(term_id).or_default();
        if doc_list.is_empty() || doc_list.last() != Some(&point_id) {
            stats.doc_freq += 1;
        }
        
        // Track document length
        *self.doc_lengths.entry(point_id).or_insert(0) += 1;
        
        // Add to posting list
        doc_list.push(point_id);
        Ok(())
    }
}
```

### MultiTermBuilder Extension

```rust
pub struct MultiTermBuilder {
    inner_builders: DashMap<u128, RwLock<TermBuilder>>,
    is_built: AtomicBool,
    
    // NEW: Per-user collection statistics
    user_collection_stats: DashMap<u128, CollectionStatisticsBuilder>,
}

struct CollectionStatisticsBuilder {
    doc_count: u64,
    sum_total_term_freq: u64,
}
```

## Scalability Considerations

| Scale Factor | Concern | Approach |
|--------------|---------|----------|
| **Many users (10K+)** | Memory for per-user stats | Lazy loading + LRU cache for BM25 statistics |
| **Many terms (1M+)** | Term stats lookup speed | Embed doc_freq in posting list header or use roaring bitmap index |
| **Long documents (10K+ tokens)** | Document norms precision | Use saturating length encoding (cap at 255) |
| **Many segments (100+)** | Cross-segment aggregation overhead | Pre-aggregate collection stats in Collection TOC |
| **High query rate (10K QPS)** | IDF computation overhead | Cache IDF per query per segment (short-lived cache) |

## Sources

- [Lucene BM25Similarity](https://lucene.apache.org/core/9_0_0/core/org/apache/lucene/search/similarities/BM25Similarity.html) - Official Lucene implementation (HIGH confidence)
- [Elasticsearch Similarity Settings](https://www.elastic.co/docs/reference/elasticsearch/index-settings/similarity) - Production BM25 configuration (HIGH confidence)
- [Wikipedia: Okapi BM25](https://en.wikipedia.org/wiki/Okapi_BM25) - BM25 formula and variants (MEDIUM confidence)
- [Lucene CollectionStatistics](https://lucene.apache.org/core/9_0_0/core/org/apache/lucene/search/CollectionStatistics.html) - Stats structure design (HIGH confidence)
- [Lucene TermStatistics](https://lucene.apache.org/core/9_0_0/core/org/apache/lucene/search/TermStatistics.html) - Term-level stats design (HIGH confidence)
- MuopDB codebase analysis - `multi_terms` module, segment architecture (HIGH confidence)
