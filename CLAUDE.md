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
