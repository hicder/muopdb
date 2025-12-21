# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MuopDB is a multi-user vector database for AI memories. It supports multiple users with isolated vector indices within the same collection. Key features:
- **Index types**: HNSW, IVF, SPANN, Multi-user SPANN (all on-disk with mmap)
- **Quantization**: Product Quantization (PQ) for compression
- **Storage**: Memory-mapped files, Write-Ahead Log (WAL), segment-based architecture
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

**Prerequisites** (MacOS with Homebrew):
```bash
brew install hdf5 protobuf openblas
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

```
Spann<Q>
├── centroids: Hnsw<NoQuantizer>  # Coarse-level search (in-memory)
└── posting_lists: IvfType<Q>     # Fine-grained search (mmap'd)
```

### Multi-User Support

`MultiSpannIndex<Q>` uses `DashMap<u128, Arc<Spann<Q>>>` for per-user indices with lazy loading.

### Quantizer Trait

```rust
pub trait Quantizer {
    type QuantizedT: VectorT<Self>;
    fn quantize(&self, value: &[f32]) -> Vec<Self::QuantizedT>;
    fn distance(&self, query: &[Self::QuantizedT], point: &[Self::QuantizedT]) -> f32;
}
```

## Important Files

- `rs/index/src/collection/core.rs`: Collection implementation (versioning, segments, WAL)
- `rs/index/src/multi_spann/index.rs`: Multi-user SPANN index
- `rs/index/src/spann/index.rs`: SPANN index
- `rs/index/src/wal/mod.rs`: Write-Ahead Log with group commit
- `rs/index/src/segment/mod.rs`: Segment trait definitions

## Code Conventions

- Uses `ouroboros` for self-referential structs
- Uses `dashmap` for concurrent segment access
- Uses `parking_lot` mutexes (not std)
- Uses `anyhow` for error handling
- Uses `rkyv` for serialization (zero-copy deserialization)
- Nightly Rust required (features: `auto_traits`, `min_specialization`)
