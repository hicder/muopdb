# Requirements: BM25 Support for MuopDB

**Defined:** 2026-04-05
**Core Value:** Users can search documents using BM25-ranked text relevance, enabling accurate keyword-based retrieval that complements vector similarity search for hybrid retrieval workflows.

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Core Algorithm

- [x] **CORE-01**: BM25 scoring formula implementation with configurable k1 (term saturation, default 1.2) and b (length normalization, default 0.75) parameters
- [x] **CORE-02**: IDF (Inverse Document Frequency) computation with flooring at 0 to prevent negative scores for high-frequency terms
- [x] **CORE-03**: TF (Term Frequency) saturation component with precision-safe formula to avoid floating-point loss
- [x] **CORE-04**: Document length normalization using configurable length definition (total terms)

### Statistics Collection

- [ ] **STAT-01**: Document frequency (DF) statistics per term collected during index building
- [ ] **STAT-02**: Term frequency (TF) statistics per document collected during index building
- [ ] **STAT-03**: Document length tracking per document stored in segment
- [x] **STAT-04**: Average document length (avgdl) computed per segment
- [x] **STAT-05**: Collection-level statistics (total doc count, total term count) stored with segment metadata

### Index Building

- [x] **BLD-01**: Tokenization pipeline using unicode-segmentation for word boundaries
- [x] **BLD-02**: Stemming integration using existing rust-stemmers crate
- [x] **BLD-03**: Stop word filtering support (optional, configurable)
- [x] **BLD-04**: Statistics serialized using rkyv for consistency with existing posting list format
- [x] **BLD-05**: Per-segment BM25 statistics stored in dedicated subdirectory (bm25_stats/)

### Query Scoring

- [ ] **QRY-01**: Single-term BM25 scoring returning score for query term against candidate documents
- [ ] **QRY-02**: Multi-term query scoring with score aggregation across query terms
- [ ] **QRY-03**: Top-K retrieval returning highest-scoring documents for text query
- [ ] **QRY-04**: Cross-segment score aggregation for multi-segment collections

### Hybrid Search Integration

- [x] **HYB-01**: Score normalization for BM25 scores (min-max normalization to [0,1] range)
- [x] **HYB-02**: Linear combination fusion with configurable alpha parameter (alpha * BM25 + (1-alpha) * vector)
- [x] **HYB-03**: Combined top-K retrieval ranking documents by hybrid score
- [x] **HYB-04**: Per-user hybrid search maintaining multi-user isolation

### API & Integration

- [x] **API-01**: BM25Config struct exposing k1 and b parameters with validation
- [ ] **API-02**: BM25Scorer component integrated with existing MultiTermIndex
- [x] **API-03**: Query API extension supporting text-only, vector-only, and hybrid queries
- [x] **API-04**: gRPC message extensions for BM25 query parameters

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Advanced BM25

- **ADV-01**: BM25F (field-weighted variant) — requires significant schema changes for multi-field documents
- **ADV-02**: BM25+ (lower-bounded IDF variant) — optimization, not essential for MVP
- **ADV-03**: k3 parameter for query term frequency — advanced query handling

### Query Features

- **QRY-05**: Phrase queries with position scoring — increases index size 3-5x
- **QRY-06**: Complex query language (AND/OR/NOT operators) — requires query parser
- **QRY-07**: Reciprocal Rank Fusion (RRF) as alternative to linear combination — needs benchmarking

### Real-time Updates

- **RT-01**: Real-time BM25 index updates for mutable segments — requires handling avgdl drift

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Full-text query language | Complex query parser not needed for MVP — simple bag-of-words sufficient |
| BM25F multi-field scoring | Requires document schema with field definitions — significant scope increase |
| Real-time mutable segment scoring | Initial implementation for immutable segments only — mutable support in v2 |
| Position-based phrase scoring | Index size increase not justified for initial release |
| Tantivy integration | Architectural impedance mismatch with segment-based storage — custom implementation preferred |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| CORE-01 | Phase 1 | Complete |
| CORE-02 | Phase 1 | Complete |
| CORE-03 | Phase 1 | Complete |
| CORE-04 | Phase 1 | Complete |
| STAT-01 | Phase 2 | Pending |
| STAT-02 | Phase 2 | Pending |
| STAT-03 | Phase 2 | Pending |
| STAT-04 | Phase 2 | Complete |
| STAT-05 | Phase 2 | Complete |
| BLD-01 | Phase 2 | Complete |
| BLD-02 | Phase 2 | Complete |
| BLD-03 | Phase 2 | Complete |
| BLD-04 | Phase 2 | Complete |
| BLD-05 | Phase 2 | Complete |
| QRY-01 | Phase 3 | Pending |
| QRY-02 | Phase 3 | Pending |
| QRY-03 | Phase 3 | Pending |
| QRY-04 | Phase 3 | Pending |
| HYB-01 | Phase 4 | Complete |
| HYB-02 | Phase 4 | Complete |
| HYB-03 | Phase 4 | Complete |
| HYB-04 | Phase 4 | Complete |
| API-01 | Phase 1 | Complete |
| API-02 | Phase 3 | Pending |
| API-03 | Phase 4 | Complete |
| API-04 | Phase 4 | Complete |

**Coverage:**
- v1 requirements: 26 total
- Mapped to phases: 26
- Unmapped: 0 ✓

---
*Requirements defined: 2026-04-05*
*Last updated: 2026-04-05 after initial definition*
