# Phase 2: Statistics & Index Building - Research

**Researched:** 2026-04-05
**Domain:** BM25 Statistics Collection, Tokenization, and rkyv Serialization
**Confidence:** HIGH

## Summary

This phase implements statistics collection and index building for BM25 text search in MuopDB. The core work involves extending the existing `MultiTermBuilder` to collect document frequency (DF), term frequency (TF), and document length statistics during index building. The implementation integrates with the existing segment flush lifecycle, stores statistics in a dedicated `bm25_stats/` subdirectory, and uses rkyv for zero-copy serialization (consistent with existing posting list format).

**Key architectural insight:** BM25 statistics must be collected during the document insertion phase in `MutableSegment::insert_for_user()` before tokenization, then persisted during segment flush alongside the existing term index. The statistics storage follows the per-segment, per-user pattern established by `MultiTermBuilder`/`MultiTermWriter`.

## Standard Stack

### Core Libraries
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `unicode-segmentation` | ^1.11 | Word boundary tokenization | Standard Rust Unicode segmentation |
| `rust-stemmers` | 1.2 | Snowball stemming | Already in codebase (StemmingTokenizer) |
| `rkyv` | 0.8.10 | Zero-copy serialization | Used by WAL and existing serialization |
| `dashmap` | 6.1.0 | Concurrent hash map | Existing pattern for multi-user indices |

### Verification (run before implementation):
```bash
cargo add unicode-segmentation --version 1.11.0  # Verify latest
```

### Installation:
```bash
# Add to rs/index/Cargo.toml:
# unicode-segmentation = "1.11"
```

## Architecture Patterns

### 1. Statistics Collection During Insertion

**What:** Extend `MultiTermBuilder` to collect DF/TF/document length stats alongside posting lists.

**When to use:** During document insertion in `MutableSegment::insert_for_user()`.

**Pattern from existing code:**
```rust
// From rs/index/src/mutable_segment.rs lines 71-105
// Process document attributes if present
for (attr_name, attr_value) in document_attribute.value {
    match attr_value.value {
        Some(proto::muopdb::attribute_value::Value::TextValue(text)) => {
            let language = /* ... language detection ... */;
            let stemming_tokenizer = StemmingTokenizer::for_language(language);
            let mut token_stream = stemming_tokenizer.input(&text);

            // Process each token and insert with the term builder
            while let Some(token) = token_stream.next() {
                let term = format!("{}:{}", attr_name, token.text);
                self.multi_term_builder.add(user_id, point_id, term)?;
            }
        }
        // ... keyword handling
    }
}
```

**BM25 Stats Extension:**
```rust
// Statistics to collect per document:
pub struct DocumentStats {
    pub doc_id: u32,              // point_id
    pub term_count: u32,          // dl (document length for avgdl)
    pub term_frequencies: HashMap<String, u32>,  // per-term TF
}

// Per-user collection
pub struct UserBm25Stats {
    pub doc_stats: HashMap<u32, DocumentStats>,  // doc_id -> stats
    pub doc_frequencies: HashMap<String, u32>,   // term -> DF
    pub total_term_count: u64,                   // total terms in segment
}
```

### 2. Tokenization Pipeline with unicode-segmentation

**What:** Replace or extend `WhiteSpaceTokenizer` to use Unicode word boundaries.

**When to use:** When tokenizing text attributes for BM25 indexing.

**Reference Implementation:**
```rust
// New file: rs/index/src/tokenizer/unicode_tokenizer.rs
use unicode_segmentation::UnicodeSegmentation;
use super::{Token, TokenStream, Tokenizer};

pub struct UnicodeTokenizer;

impl Tokenizer for UnicodeTokenizer {
    type TokenStream<'a> = UnicodeTokenStream<'a>;

    fn input<'a>(&self, text: &'a str) -> Self::TokenStream<'a> {
        UnicodeTokenStream {
            graphemes: text.unicode_word_indices(),
            token: Token::new(String::new()),
        }
    }
}

pub struct UnicodeTokenStream<'a> {
    graphemes: unicode_segmentation::UnicodeWordIndices<'a>,
    token: Token,
}

impl<'a> TokenStream for UnicodeTokenStream<'a> {
    fn advance(&mut self) -> bool {
        if let Some((_, word)) = self.graphemes.next() {
            self.token = Token::new(word.to_lowercase());
            true
        } else {
            false
        }
    }

    fn token(&self) -> Token {
        self.token.clone()
    }

    fn next(&mut self) -> Option<Token> {
        if self.advance() { Some(self.token.clone()) } else { None }
    }
}
```

**Stop Word Filtering Pattern:**
```rust
pub struct StopWordFilter<T: TokenStream> {
    inner: T,
    stop_words: HashSet<String>,
}

impl<T: TokenStream> TokenStream for StopWordFilter<T> {
    fn advance(&mut self) -> bool {
        while self.inner.advance() {
            let token = self.inner.token();
            if !self.stop_words.contains(&token.text) {
                return true;
            }
        }
        false
    }
    // ... token() and next()
}
```

### 3. Document Norm Encoding Strategies

**What:** Store document lengths efficiently for query-time length normalization.

**Options Considered:**

| Strategy | Storage | Pros | Cons |
|----------|---------|------|------|
| **Raw u32 array** | 4 bytes/doc | Simple, fast | Larger for small segments |
| **Delta encoding + Elias-Fano** | Variable | Compact for sorted docs | Complex, slower |
| **Block-compressed u32** | ~1 byte/doc | Good compression, fast | Block size tuning needed |

**Recommended:** Raw u32 array indexed by `point_id` for simplicity and zero-copy access. Document lengths are random-accessed during scoring, so compression would hurt performance.

**Layout:**
```rust
// In bm25_stats/ subdirectory
// doc_norms.bin: [u32; num_docs] - document lengths
// term_stats.bin: [TermStats] - per-term DF
// header.json: { num_docs, total_terms, avgdl }
```

### 4. rkyv Serialization Pattern

**What:** Use rkyv for consistent, zero-copy deserialization (per D-08).

**Source:** WAL file pattern in `rs/index/src/wal/file.rs` lines 8, 285-296

```rust
use rkyv::util::AlignedVec;
use rkyv::{Archive, Deserialize, Serialize};

// Example stats structure for rkyv
#[derive(Archive, Serialize, Deserialize, Debug, Clone)]
pub struct Bm25SegmentStats {
    pub num_docs: u32,
    pub total_terms: u64,
    pub avgdl: f64,
}

#[derive(Archive, Serialize, Deserialize, Debug, Clone)]
pub struct TermStat {
    pub term_id: u64,
    pub doc_freq: u32,  // DF
}

// Serialization
let stats = Bm25SegmentStats { /* ... */ };
let serialized = rkyv::to_bytes::<_, 256>(&stats).unwrap();
std::fs::write("stats.rkyv", &serialized)?;

// Deserialization (zero-copy)
let data = std::fs::read("stats.rkyv")?;
let aligned = AlignedVec::<16>::from(data);
let archived = rkyv::access::<ArchivedBm25SegmentStats, 16>(&aligned).unwrap();
```

**Important:** Use `AlignedVec` with 16-byte alignment (matching WAL pattern) to ensure proper alignment for rkyv's zero-copy access.

### 5. Segment Flush Integration

**What:** Hook into `MutableSegment::build()` to write BM25 stats alongside terms.

**Source:** `rs/index/src/segment/mutable_segment.rs` lines 124-150

```rust
pub fn build(&mut self, base_directory: String, name: String) -> Result<()> {
    // ... existing setup ...

    // Build SPANN
    self.multi_spann_builder.build()?;
    let multi_spann_writer = MultiSpannWriter::new(segment_directory.clone());
    multi_spann_writer.write(&mut self.multi_spann_builder)?;

    // Build terms
    self.multi_term_builder.build()?;
    let multi_term_writer = MultiTermWriter::new_with_segment_dir(segment_directory.clone());
    multi_term_writer.write(&self.multi_term_builder)?;

    // NEW: Write BM25 statistics
    let bm25_stats_writer = Bm25StatsWriter::new(segment_directory.clone());
    bm25_stats_writer.write(&self.multi_term_builder)?;  // Extract stats from builder

    // ... cleanup ...
}
```

### 6. Multi-User Stats Isolation

**What:** Per-user statistics following `DashMap<u128, RwLock<TermBuilder>>` pattern.

**Source:** `rs/index/src/multi_terms/builder.rs` lines 10-15

```rust
pub struct MultiBm25StatsBuilder {
    inner_stats: DashMap<u128, RwLock<UserBm25Stats>>,
}

impl MultiBm25StatsBuilder {
    pub fn add_document(&self, user_id: u128, doc_id: u32, text: &str) -> Result<()> {
        let stats = self.inner_stats
            .entry(user_id)
            .or_insert_with(|| RwLock::new(UserBm25Stats::new()));
        
        let mut guard = stats.write().unwrap();
        guard.add_document(doc_id, text)?;
        Ok(())
    }
}
```

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Unicode word segmentation | Custom regex | `unicode-segmentation` | Handles all Unicode scripts correctly |
| Stop word lists | Hardcoded list | Configurable with common lists | English, French, etc. stop words differ |
| Term normalization | Manual lowercasing | StemmingTokenizer pattern | Consistent with existing code |
| Stats serialization | Custom binary format | rkyv | Zero-copy, consistent with WAL/posting lists |
| Document length storage | Compressed per doc | u32 array indexed by point_id | Fast random access for scoring |

## Common Pitfalls

### Pitfall 1: Document Length Definition Ambiguity
**What goes wrong:** Different interpretations of "document length" (bytes, chars, tokens) cause inconsistent scoring.

**Why it happens:** BM25 formula requires `dl` and `avgdl` to use same unit.

**How to avoid:** 
- Define document length = number of tokens after tokenization (per D-01)
- Store this explicitly, don't recompute at query time

### Pitfall 2: Tokenization Mismatch Between Index and Query
**What goes wrong:** Query tokenizes differently than indexing, causing term lookup failures.

**Why it happens:** Different tokenizers or preprocessing between phases.

**How to avoid:**
- Use same `UnicodeTokenizer` + `StemmingTokenizer` + `StopWordFilter` pipeline
- Store tokenizer config in segment metadata

### Pitfall 3: Per-User DF Counting Errors
**What goes wrong:** Computing DF across all users instead of per-user breaks isolation.

**Why it happens:** BM25 IDF should only consider user's own documents for privacy.

**How to avoid:**
- Store per-user DF counts in separate maps (following MultiTermBuilder pattern)
- Never mix statistics across user boundaries

### Pitfall 4: Mutable Segment avgdl Drift
**What goes wrong:** Real-time updates to mutable segments change avgdl, invalidating cached scores.

**Why it happens:** Mutable segments accept new documents continuously.

**Warning signs:** Scores vary without reindexing.

**How to avoid:**
- Only compute BM25 scores for immutable segments (per Out of Scope: RT-01)
- Document this limitation; mutable segments use simpler filtering only

### Pitfall 5: rkyv Alignment Issues
**What goes wrong:** Unaligned memory access when deserializing rkyv data.

**Why it happens:** rkyv requires specific alignment for zero-copy access.

**How to avoid:**
- Always use `AlignedVec<16>` as in WAL pattern
- Never use raw `Vec<u8>` for rkyv data

## Code Examples

### BM25 Stats Collection Integration

```rust
// rs/index/src/multi_terms/builder.rs extension

use std::collections::HashMap;

/// Per-document statistics for BM25
#[derive(Debug, Clone, Default)]
pub struct DocStats {
    pub term_count: u32,
    pub term_freqs: HashMap<String, u32>,  // term -> frequency in this doc
}

impl MultiTermBuilder {
    /// Add a document with BM25 stats tracking
    pub fn add_with_stats(
        &self,
        user_id: u128,
        point_id: u32,
        terms: Vec<String>,  // Pre-tokenized terms
    ) -> Result<()> {
        // Existing: add to posting lists
        for term in &terms {
            self.add(user_id, point_id, term.clone())?;
        }
        
        // NEW: collect statistics
        let term_freqs = Self::compute_term_freqs(&terms);
        let doc_stats = DocStats {
            term_count: terms.len() as u32,
            term_freqs,
        };
        
        // Store per user, per doc
        self.bm25_stats
            .entry(user_id)
            .or_insert_with(|| RwLock::new(UserBm25Stats::new()))
            .write()
            .unwrap()
            .add_doc_stats(point_id, doc_stats);
        
        Ok(())
    }
    
    fn compute_term_freqs(terms: &[String]) -> HashMap<String, u32> {
        let mut freqs = HashMap::new();
        for term in terms {
            *freqs.entry(term.clone()).or_insert(0) += 1;
        }
        freqs
    }
}
```

### Tokenizer Chain Pattern

```rust
// rs/index/src/tokenizer/mod.rs - pipeline composition

pub struct TokenizerPipeline {
    unicode: UnicodeTokenizer,
    stemmer: StemmingTokenizer,
    stop_words: HashSet<String>,
}

impl TokenizerPipeline {
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        
        // Step 1: Unicode word segmentation
        let mut stream = self.unicode.input(text);
        while let Some(token) = stream.next() {
            // Step 2: Stemming
            let stemmed = self.stemmer.stem(&token.text);
            
            // Step 3: Stop word filtering
            if !self.stop_words.contains(&stemmed) {
                tokens.push(stemmed);
            }
        }
        
        tokens
    }
}
```

### Stats Persistence Pattern

```rust
// rs/index/src/bm25/stats_writer.rs

use rkyv::util::AlignedVec;
use rkyv::Archive;

pub struct Bm25StatsWriter {
    base_directory: String,
}

impl Bm25StatsWriter {
    pub fn write(&self, builder: &MultiTermBuilder) -> Result<()> {
        // Create bm25_stats subdirectory
        let stats_dir = format!("{}/bm25_stats", self.base_directory);
        std::fs::create_dir_all(&stats_dir)?;
        
        for user_entry in builder.user_stats_iter() {
            let user_id = user_entry.key();
            let stats = user_entry.value().read().unwrap();
            
            // Write document norms
            let norms_path = format!("{}/{}.norms", stats_dir, user_id);
            self.write_doc_norms(&norms_path, &stats.doc_norms)?;
            
            // Write term stats (DF per term)
            let terms_path = format!("{}/{}.terms", stats_dir, user_id);
            self.write_term_stats(&terms_path, &stats.term_stats)?;
            
            // Write segment metadata (avgdl, num_docs)
            let meta_path = format!("{}/{}.meta", stats_dir, user_id);
            self.write_segment_meta(&meta_path, &stats.segment_meta)?;
        }
        
        Ok(())
    }
    
    fn write_doc_norms(&self, path: &str, norms: &[u32]) -> Result<()> {
        // Simple array of u32: [doc_len_0, doc_len_1, ...]
        // Indexed by point_id
        let data = unsafe {
            std::slice::from_raw_parts(
                norms.as_ptr() as *const u8,
                norms.len() * 4
            )
        };
        std::fs::write(path, data)?;
        Ok(())
    }
    
    fn write_term_stats(&self, path: &str, term_stats: &HashMap<String, TermStat>) -> Result<()> {
        // Serialize with rkyv
        let aligned = rkyv::to_bytes::<_, 256>(term_stats)
            .map_err(|e| anyhow::anyhow!("rkyv serialization failed: {}", e))?;
        std::fs::write(path, &aligned)?;
        Ok(())
    }
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Whitespace tokenization | Unicode segmentation | Phase 2 | Better multilingual support |
| Stemming only | Stemming + stop words | Phase 2 | Reduces noise from common words |
| No document length tracking | Explicit norm storage | Phase 2 | Enables proper BM25 length normalization |
| serde_json serialization | rkyv serialization | Phase 2 | Zero-copy access, consistent with WAL |

**No deprecated patterns** - this is new functionality.

## Open Questions

1. **Stop word list source**
   - What we know: Need configurable stop word lists
   - What's unclear: Use built-in lists or external file? Which languages?
   - Recommendation: Start with English stop words (common list), make configurable via BM25Config

2. **Document length for empty documents**
   - What we know: BM25 formula needs dl > 0 to avoid division by zero
   - What's unclear: Should empty docs have dl=0 or dl=1?
   - Recommendation: dl=0 is fine; BM25Scorer already has `avgdl.max(1.0)` protection

3. **Multi-field document support (out of scope per ADV-01)**
   - What we know: BM25F (field-weighted) is deferred to v2
   - What's unclear: How to store per-field lengths for future BM25F?
   - Recommendation: Store total dl only; BM25F will require schema changes anyway

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|-------------|-----------|---------|----------|
| unicode-segmentation | Tokenization | ✗ | — | Add to Cargo.toml |
| rust-stemmers | Stemming | ✓ | 1.2 | — |
| rkyv | Serialization | ✓ | 0.8.10 | — |

**Missing dependencies with no fallback:**
- unicode-segmentation must be added to `rs/index/Cargo.toml`

## Sources

### Primary (HIGH confidence)
- `rs/index/src/multi_terms/builder.rs` - MultiTermBuilder patterns
- `rs/index/src/multi_terms/writer.rs` - Term index serialization
- `rs/index/src/segment/mutable_segment.rs` - Segment flush lifecycle
- `rs/index/src/wal/file.rs` - rkyv serialization patterns (lines 8, 285-296)
- `rs/index/src/tokenizer/stemming_tokenizer.rs` - TokenStream trait implementation
- `.planning/phases/01-core-algorithm-config/01-CONTEXT.md` - Phase 1 decisions (D-01 to D-08)
- `.planning/REQUIREMENTS.md` - STAT-01 to STAT-05, BLD-01 to BLD-05

### Secondary (MEDIUM confidence)
- Unicode Segmentation crate documentation (https://docs.rs/unicode-segmentation)
- BM25 algorithm references from Phase 1 context (Lucene, Wikipedia)

### Tertiary (LOW confidence)
- None - all findings verified with code inspection

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - unicode-segmentation is de facto standard for Rust Unicode handling
- Architecture: HIGH - clear patterns exist in MultiTermBuilder/MultiTermWriter
- Pitfalls: MEDIUM-HIGH - based on BM25 literature and codebase patterns

**Research date:** 2026-04-05
**Valid until:** 2026-05-05 (30 days for stable dependencies)
