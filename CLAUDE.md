# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MuopDB is a multi-user vector database for AI memories. It supports multiple users with isolated vector indices within the same collection. Key features:
- **Index types**: HNSW, IVF, SPANN, Multi-user SPANN (on-disk with async block-based storage)
- **Quantization**: Product Quantization (PQ) for compression
- **Storage**: Async block-based I/O with optional io_uring (Linux), Write-Ahead Log (WAL), segment-based architecture
- **Distributed**: Aggregator for sharding and query fan-out

## Build & Test Commands

```bash
# Build release
cargo build --release

# Run all tests
cargo test --release

# Run specific test
cargo test --release -p index -- test_name

# Run benchmarks
cargo bench -p benchmarks

# Run lints
cargo clippy --all-targets --all-features -- -D warnings
```

**Prerequisites**:

MacOS (Homebrew):
```bash
brew install protobuf openblas
```

Linux (Arch-based):
```bash
sudo pacman -Syu protobuf openblas
```

Linux (Debian-based):
```bash
sudo apt-get install libprotobuf-dev libopenblas-dev
```

## Core Architecture

### Workspace Structure (`rs/`)

| Crate | Purpose |
|-------|---------|
| `index/` | Core index implementations (SPANN, HNSW, IVF) and collection management |
| `index_server/` | gRPC server handling client requests |
| `index_writer/` | Offline index building tool |
| `aggregator/` | Distributed query aggregation (shard routing) |
| `quantization/` | Vector quantization (ProductQuantizer, NoQuantizer) |
| `compression/` | Compression utilities (Elias-Fano, NOC) |
| `config/` | Configuration types |
| `utils/` | Shared utilities (distance, bloom filter, KMeans) |
| `proto/` | gRPC protocol definitions |
| `metrics/` | Internal metrics |
| `cli/` | Command-line interface |
| `demo/` | Demo binaries for testing (insert, search) |

### Key Data Flow

**Insert path**: gRPC → `Collection::write_to_wal()` → WAL (group commit) → `MutableSegment::insert_for_user()` → `MultiSpannBuilder::insert()`

**Search path**: gRPC → `Collection::get_snapshot()` → parallel segment search → `MultiSpannIndex::search_for_user()` → `Spann::search()` (HNSW centroids → IVF posting lists)

### Segment Architecture

```
Collection
├── MutableSegment (in-memory, writable) → flushed periodically
├── PendingMutableSegment (during flush/build)
└── ImmutableSegments (read-only, mmap'd) → merged by optimizers
```

### SPANN Index Structure

The current SPANN implementation uses block-based async storage:

```
Spann<Q>
├── centroids: BlockBasedHnsw<NoQuantizer>  # Coarse-level search
└── posting_lists: BlockBasedIvf<Q>         # Fine-grained search
```

The codebase previously supported mmap-based implementations but has migrated to block-based storage for better cache locality and async I/O support.

### Multi-User Support

`MultiSpannIndex<Q>` uses `DashMap<u128, Arc<Spann<Q>>>` for per-user indices with lazy loading.

### Hybrid Search (MultiTerms)

The `multi_terms` module supports combining text and vector search:
- `MultiTermIndex`: Combined term + vector index per segment
- Supports filtering and scoring across both modalities

### Env Abstraction (Storage Layer)

The `Env` trait (`rs/utils/src/file_io/env.rs`) provides a unified interface for different storage backends:

- `FileType::MMap`: Traditional memory-mapped files
- `FileType::CachedStandard`: Async file I/O with block cache (cross-platform)
- `FileType::CachedIoUring`: Async file I/O with io_uring (Linux only, better performance)

```rust
pub trait Env: Send + Sync {
    async fn open(&self, path: &str) -> Result<OpenResult>;
    async fn open_append(&self, path: &str) -> Result<OpenAppendResult>;
    async fn close(&self, file_id: FileId) -> Result<()>;
}
```

The `DefaultEnv` implementation manages a `BlockCache` for efficient I/O operations.

### Quantizer Trait

```rust
pub trait Quantizer: Send + Sync {
    type QuantizedT: VectorT<Self> + Send + Sync;
    fn quantize(&self, value: &[f32]) -> Vec<Self::QuantizedT>;
    fn quantized_dimension(&self) -> usize;
    fn distance(&self, query: &[Self::QuantizedT], point: &[Self::QuantizedT], implem: L2DistanceCalculatorImpl) -> f32;
}
```

## Important Files

- `rs/index/src/collection/core.rs`: Collection implementation (versioning, segments, WAL)
- `rs/index/src/multi_spann/index.rs`: Multi-user SPANN index
- `rs/index/src/spann/index.rs`: SPANN index with BlockBasedHnsw and BlockBasedIvf
- `rs/index/src/hnsw/block_based/index.rs`: Block-based HNSW implementation
- `rs/index/src/ivf/block_based/index.rs`: Block-based IVF implementation
- `rs/index/src/wal/mod.rs`: Write-Ahead Log with group commit
- `rs/index/src/segment/mod.rs`: Segment trait definitions
- `rs/utils/src/file_io/env.rs`: Storage abstraction layer (Env trait)

## Code Conventions

- Uses `ouroboros` for self-referential structs
- Uses `dashmap` for concurrent segment access
- Uses `async_lock` for mutexes (not std)
- Uses `anyhow` for error handling
- Uses `rkyv` for serialization (zero-copy deserialization)
- Uses `tokio` for async runtime
- Uses `tonic` for gRPC server/client
- Nightly Rust required (features: `auto_traits`, `min_specialization`, `portable_simd`)

## Key Data Types

- **User/Doc IDs**: `u128` (high 64 bits + low 64 bits for sharding)
- **Vector IDs**: Internal mapping uses `u128` to external indices
- **Collection versioning**: Atomic counter for snapshot isolation

<!-- GSD:project-start source:PROJECT.md -->
## Project

**BM25 Support for MuopDB**

Add BM25 (Best Matching 25) ranking algorithm to MuopDB's text search capabilities. BM25 is a probabilistic ranking function used by search engines to rank documents by relevance to a given search query. This will enable hybrid search combining BM25 text relevance scores with vector similarity scores.

The implementation will integrate with the existing `multi_terms` module, extending the `TermIndex` infrastructure to support document frequency statistics, term frequency saturation, and length normalization - the core components of BM25 scoring.

**Core Value:** Users can search documents using BM25-ranked text relevance, enabling accurate keyword-based retrieval that complements vector similarity search for hybrid retrieval workflows.

### Constraints

- **Architecture**: Must integrate with existing segment-based architecture (mutable/immutable segments)
- **Storage**: Statistics must persist with segment files for recovery
- **Concurrency**: Must work with existing DashMap-based multi-user indices
- **Performance**: Scoring should not significantly impact query latency
<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->
## Technology Stack

## Runtime & Async
- `tokio` v1.24 with multi-threaded runtime, sync, signal, and macro features
- `futures` v0.3.31 for async utilities
- `async-trait` v0.1 for async trait definitions
- `async-lock` v3.4 for async synchronization primitives
- `dashmap` v6.1.0 - Concurrent hash map for multi-user indices
- `parking_lot` v0.12.3 - Fast mutexes (with `send_guard` feature)
- `crossbeam` v0.8 - Lock-free data structures and channels
- `rayon` v1.10.0 - Data parallelism for CPU-intensive operations
- `ouroboros` v0.18.5 - Used extensively for self-referential types in indices
- `atomic_refcell` v0.1.13 - Atomic cell types
## Serialization & RPC
- `rkyv` v0.8.10 - Zero-copy deserialization for index data
- `serde` v1.0 + `serde_json` + `serde_yaml` - General serialization
- `prost` v0.13.4 - Protocol Buffers code generation
- `byteorder` v1.5.0 - Byte order handling for packed data
- `tonic` v0.12.3 - gRPC client/server framework
- `tonic-build` v0.12.3 - Build-time code generation
- `tonic-reflection` v0.12.3 - gRPC server reflection support
- `tokio-stream` v0.1 - Streaming for gRPC
## Storage & I/O
- **Async block-based I/O** via custom `Env` trait in `rs/utils/src/file_io/env.rs`
- `memmap2` v0.9.5 - Memory-mapped file support (legacy path)
- `moka` v0.12 - Concurrent cache with futures support
- `nix` v0.29 with `fs` features - Unix-specific file operations
- **Linux-only:** `io-uring` v0.6 + `tokio-uring` v0.5 for high-performance async I/O
- **Standard:** `CachedStandard` - Cross-platform async file I/O with block cache
- `object_store` v0.13.1 with AWS features - S3-compatible storage backend
## Vector Processing
- `ndarray` v0.16.1 - N-dimensional arrays
- `ndarray-linalg` v0.17.0 with `openblas-system` - Linear algebra operations
- `ndarray-rand` v0.15.0 - Random array generation
- `kmeans` v0.11.0 - K-means clustering for vector quantization
- `num-traits` v0.2.19 - Numeric trait abstractions
- `ordered-float` v4.3.0 - Ordered floating-point types
## Quantization & Compression
- `quantization` crate (internal) - Product Quantization (PQ) and NoQuantizer implementations
- `compression` crate (internal) - Elias-Fano encoding, NOC (no compression)
- `integer-encoding` v4.0.2 - Variable-length integer encoding
- `bitvec` v1.0 - Bit vector operations
- `bit-vec` v0.8.0 - Alternative bit vector implementation
- `roaring` v0.10.6 - Compressed bitmaps for posting lists
## Observability
- `log` v0.4.22 - Standard logging facade
- `env_logger` v0.11.5 - Environment-based logger
- `tracing` v0.1 - Structured logging and instrumentation
- `tracing-subscriber` v0.3 with `env-filter` and `json` features
- `opentelemetry` v0.27 + `opentelemetry_sdk` with tokio runtime
- `opentelemetry-otlp` v0.27 with tonic - OTLP exporter
- `tracing-opentelemetry` v0.28 - Tracing integration
- `prometheus-client` v0.23.1 - Prometheus metrics
- `pprof` v0.15 with `prost-codec` features - CPU profiling
## Data Structures
- `dashmap` v6.1.0 - Concurrent hash map (used for multi-user indices)
- `sorted-vec` v0.8.5 - Sorted vector implementation
- `odht` v0.3.1 - Hash table implementation
- `xxhash-rust` v0.8.15 with `xxh3` - Fast hashing
- `rust-stemmers` v1.2 - Text stemming for full-text search
- `rand` v0.8.5 - Random number generation
## Build System
- Cargo workspace with 14+ crates
- Rust Edition 2021 (most crates), Edition 2024 (metrics crate)
- Resolver version 2
- Nightly Rust required (features: `auto_traits`, `min_specialization`, `portable_simd`)
- `protobuf` (protoc compiler) - For gRPC code generation
- `openblas` - Linear algebra library for ndarray-linalg
- `criterion` v0.4 - Benchmark framework
- `tempdir` v0.3.7 - Temporary directories for tests
## Key Libraries
| Library | Purpose | Version |
|---------|---------|---------|
| `tokio` | Async runtime | 1.24 |
| `tonic` | gRPC framework | 0.12.3 |
| `prost` | Protocol Buffers | 0.13.4 |
| `rkyv` | Zero-copy serialization | 0.8.10 |
| `dashmap` | Concurrent hash map | 6.1.0 |
| `ouroboros` | Self-referential structs | 0.18.5 |
| `ndarray` | N-dimensional arrays | 0.16.1 |
| `rayon` | Data parallelism | 1.10.0 |
| `moka` | Async cache | 0.12 |
| `memmap2` | Memory-mapped files | 0.9.5 |
| `object_store` | S3-compatible storage | 0.13.1 |
| `opentelemetry` | Distributed tracing | 0.27 |
| `tracing` | Structured logging | 0.1 |
| `io-uring` | Linux async I/O | 0.6 |
| `tokio-uring` | Tokio io_uring integration | 0.5 |
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

## Style Guidelines
- Uses `rustfmt` with configuration in `rustfmt.toml`:
- CI enforces formatting via `cargo fmt -- --check`
- **Files:** `snake_case.rs` (e.g., `core.rs`, `graph_storage.rs`)
- **Directories:** `snake_case` (e.g., `block_based/`, `multi_spann/`)
- **Structs/Enums:** `CamelCase` (e.g., `HnswBuilder`, `WalOpType`)
- **Functions/Methods:** `snake_case` (e.g., `reindex()`, `get_doc_id()`)
- **Variables:** `snake_case` (e.g., `max_neighbors`, `base_directory`)
- **Constants:** `SCREAMING_SNAKE_CASE` or `UPPERCASE` (e.g., `NUM_GROUPS`)
- **Type Parameters:** Single uppercase letter (e.g., `Q: Quantizer`)
- **Generic associated types:** `QuantizedT` pattern
## Error Handling
- Uses `anyhow` crate for error handling throughout
- Functions return `anyhow::Result<T>` instead of `std::result::Result<T, E>`
- Error propagation with `?` operator
- Context attachment with `.context()` method
## Async Patterns
- Uses `tokio` with features: `macros`, `rt-multi-thread`, `sync`, `signal`
- Tests use `#[tokio::test]` attribute
- **Use `async-lock` (not std sync):** For async code, always use `async_lock::RwLock`
- **Use `tokio::sync`:** For channels and oneshot: `tokio::sync::mpsc`, `tokio::sync::oneshot`, `tokio::sync::watch`
- **Use `std::sync`:** For `Arc` only (never for `Mutex`/`RwLock` in async contexts)
- Uses `tokio::fs` for file operations in async contexts
- Uses `tokio::io::{AsyncReadExt, AsyncWriteExt, AsyncSeekExt}`
## Serialization
- Uses `rkyv` for zero-copy deserialization
- Uses `rkyv::util::AlignedVec` for aligned byte storage
- Found in: `rs/index/src/wal/file.rs`, `rs/index/src/wal/entry.rs`
- Uses `serde` with `serde_json` and `serde_yaml` for config files
- `#[derive(Serialize, Deserialize, Debug, Clone)]` pattern
#[derive(Serialize, Deserialize, Debug, Clone)]
## Documentation
- Standard Rust doc comments with `///`
- Comprehensive documentation for public APIs
- `# Arguments`, `# Returns`, `# Warning` sections
#[cfg(test)]
- No per-crate README files found
- Documentation centralized in `AGENTS.md` at repository root
## Specialized Patterns
- Channel: `nightly` (specified in `rust-toolchain.toml`)
- Required features:
- Uses `ouroboros` crate for self-referential struct handling
- Uses `dashmap` for concurrent hash maps
- Uses `DashMap<u128, Arc<Spann<Q>>>` pattern for per-user indices
- Uses `tracing` crate (not `log`)
- `tracing::debug!()` for debug logging
- `tracing-subscriber` with OpenTelemetry support
- Uses `cargo clippy --all-targets --all-features -- -D warnings`
- Workspace-level lint configuration in `Cargo.toml`:
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

## High-Level Design
## Core Components
| Component | Location | Responsibility |
|-----------|----------|----------------|
| Collection | `rs/index/src/collection/core.rs` | Central coordinator managing segments, versioning, WAL, and concurrency |
| Segment | `rs/index/src/segment/mod.rs` | Trait for partial indices: mutable (in-memory), immutable (disk), pending |
| SPANN Index | `rs/index/src/spann/index.rs` | Two-level index: HNSW centroids + IVF posting lists |
| MultiSpannIndex | `rs/index/src/multi_spann/index.rs` | Per-user SPANN indices with lazy loading via DashMap |
| WAL | `rs/index/src/wal/wal.rs` | Write-ahead log with group commit for durability |
| Index Server | `rs/index_server/src/index_server.rs` | gRPC server handling insert/search/remove requests |
| Aggregator | `rs/aggregator/src/aggregator.rs` | Distributed query fan-out and result aggregation |
| Quantizer | `rs/quantization/src/quantization.rs` | Product Quantization (PQ) and NoQuantizer trait |
| Storage Env | `rs/utils/src/file_io/env.rs` | Storage abstraction: mmap, cached I/O, io_uring, object store |
| Block Cache | `rs/utils/src/block_cache/mod.rs` | Block-based caching layer for async file I/O |
## Data Flow
### Insert Path
### Search Path
### Flush Path
## Key Abstractions
### Segment Abstraction
```rust
#[async_trait::async_trait]
```
- `MutableSegment`: In-memory, writable, uses `MultiSpannBuilder`
- `ImmutableSegment`: Disk-based, read-only, uses `MultiSpannIndex`
- `PendingSegment`: In-progress merge/vacuum operations
### Storage Env Abstraction
```rust
#[async_trait]
```
### Quantizer Trait
```rust
```
### BoxedImmutableSegment
- `FinalizedSegment`: Complete immutable segment
- `PendingSegment`: Segment undergoing merge/vacuum
- `MockedNoQuantizationSegment`: Test mock
## Segment Architecture
```
```
## SPANN Index Structure
```
```
## Versioning and Concurrency
- **TableOfContent (TOC)**: Defines active segments per version
- **Atomic Versioning**: TOC files written atomically via rename
- **Snapshot Pinning**: Reference counting prevents segment deletion during search
- **Multi-layered Locking**:
## Multi-User Support
- **Lazy Loading**: User indices loaded from disk on first access
- **Shared Segments**: All users share same segment files, separate index metadata
- **Per-User Deletions**: Invalidated IDs tracked per user
## WAL and Group Commit
- **Group Commit**: Concurrent writes batched into single WAL append
- **Leader/Follower**: First writer becomes leader, others follow
- **Timeout Handling**: First follower may become leader on timeout
- **Sync Strategy**: WAL synced to disk after batch append
- **Recovery**: On init, WAL replayed to reconstruct mutable segment state
## Optimizers
- **MergeOptimizer**: Combines multiple small segments into larger ones
- **VacuumOptimizer**: Removes invalidated documents, reclaims space
- Both implement `SegmentOptimizer` trait
<!-- GSD:architecture-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd:quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd:debug` for investigation and bug fixing
- `/gsd:execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->

<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd:profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
