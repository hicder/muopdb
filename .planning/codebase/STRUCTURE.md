# Code Structure

**Analysis Date:** 2026-04-04

## Workspace Layout

```
rs/
├── aggregator/          # Distributed query aggregation (sharding)
│   └── src/
│       ├── main.rs
│       ├── aggregator.rs
│       ├── node_manager.rs
│       └── shard_manager.rs
├── benchmarks/          # Performance benchmarks
│   └── src/
│       ├── insertion.rs
│       ├── deletion_and_vacuum.rs
│       ├── vacuum.rs
│       └── wal_insertion.rs
├── cli/                 # Command-line interface and tools
│   └── src/
│       ├── main.rs           # Client binary
│       └── index_viewer.rs   # Index inspection tool
├── compression/         # Integer sequence compression
│   └── src/
│       └── lib.rs       # Elias-Fano, NOC encoding
├── config/              # Configuration types
│   └── src/
│       └── lib.rs       # CollectionConfig, SearchParams
├── demo/                # Demo binaries for testing
│   └── src/
│       └── main.rs
├── index/               # Core index implementations
│   └── src/
│       ├── lib.rs
│       ├── collection/      # Collection management
│       │   ├── core.rs      # Main Collection struct
│       │   ├── snapshot.rs  # Snapshot for consistent reads
│       │   └── mod.rs
│       ├── hnsw/            # HNSW index (Hierarchical Navigable Small World)
│       │   ├── block_based/ # Block-based async HNSW
│       │   └── builder.rs
│       ├── ivf/             # IVF index (Inverted File)
│       │   ├── block_based/ # Block-based async IVF
│       │   ├── files/       # Invalidated IDs storage
│       │   └── builder.rs
│       ├── multi_spann/     # Multi-user SPANN
│       │   ├── index.rs     # MultiSpannIndex
│       │   ├── builder.rs   # MultiSpannBuilder
│       │   ├── reader.rs
│       │   └── writer.rs
│       ├── multi_terms/     # Hybrid text + vector search
│       │   ├── index.rs
│       │   ├── builder.rs
│       │   └── writer.rs
│       ├── optimizers/      # Segment optimization (merge, vacuum)
│       │   ├── merge.rs
│       │   └── vacuum.rs
│       ├── posting_list/    # Posting list compression and storage
│       ├── query/           # Query planning and execution
│       │   ├── planner.rs
│       │   ├── async_planner.rs
│       │   └── iters/       # Query iterators
│       ├── segment/         # Segment implementations
│       │   ├── mod.rs       # Segment trait, BoxedImmutableSegment
│       │   ├── mutable_segment.rs
│       │   ├── immutable_segment.rs
│       │   ├── pending_mutable_segment.rs
│       │   └── pending_segment.rs
│       ├── spann/           # SPANN index (single-user)
│       │   ├── index.rs
│       │   ├── builder.rs
│       │   ├── reader.rs
│       │   └── writer.rs
│       ├── terms/           # Term index for hybrid search
│       ├── tokenizer/       # Text tokenization
│       ├── vector/          # Vector storage abstractions
│       ├── wal/             # Write-Ahead Log
│       │   ├── wal.rs
│       │   ├── entry.rs
│       │   └── file.rs
│       ├── traverse_state.rs
│       └── utils.rs
├── index_server/        # gRPC index server
│   └── src/
│       ├── main.rs
│       ├── index_server.rs    # IndexServer gRPC implementation
│       ├── http_server.rs     # HTTP admin endpoints
│       ├── admin_server.rs    # Admin gRPC services
│       ├── collection_manager.rs
│       ├── collection_catalog.rs
│       └── collection_provider.rs
├── index_writer/        # Offline index building tool
│   └── src/
│       ├── main.rs
│       ├── lib.rs
│       ├── index_writer.rs
│       ├── config.rs
│       ├── hnsw_reindexer.rs
│       └── input/           # Input formats (HDF5, etc.)
├── metrics/             # Internal and API metrics
│   └── src/
│       ├── lib.rs
│       ├── internal.rs
│       └── api.rs
├── proto/               # gRPC protocol definitions
│   └── src/
│       └── lib.rs       # Compiled protobuf definitions
├── quantization/        # Vector quantization
│   └── src/
│       ├── lib.rs
│       ├── quantization.rs  # Quantizer trait
│       ├── noq/             # NoQuantizer
│       ├── pq/              # ProductQuantizer
│       └── rabitq/          # RabitQ quantization
└── utils/               # Shared utilities
    └── src/
        ├── lib.rs
        ├── distance/          # Distance calculations (L2, dot product)
        ├── file_io/           # File I/O abstractions
        │   ├── env.rs         # Env trait, DefaultEnv
        │   ├── cached_file.rs
        │   ├── mmap_file.rs
        │   ├── standard_file.rs
        │   └── uring_file.rs
        ├── block_cache/       # Block-based caching
        │   ├── cache.rs
        │   └── disk_cache.rs
        ├── bloom_filter/      # Bloom filter implementations
        ├── kmeans_builder/      # K-means clustering
        └── on_disk_ordered_map/ # On-disk hash maps
```

## Crate Dependencies

```
                    ┌─────────────┐
                    │   proto     │
                    └──────┬──────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
         ▼                 ▼                 ▼
   ┌─────────┐       ┌──────────┐      ┌──────────┐
   │  config │◄─────│  utils   │─────►│ metrics  │
   └────┬────┘      └────┬─────┘      └──────────┘
        │                │
        │    ┌───────────┼───────────┐
        │    │           │           │
        ▼    ▼           ▼           ▼
   ┌─────────┐     ┌──────────┐  ┌──────────┐
   │compression│   │quantization│  │  index   │
   └─────────┘     └──────────┘  └────┬─────┘
                                     │
                    ┌────────────────┼────────────────┐
                    │                │                │
                    ▼                ▼                ▼
              ┌──────────┐    ┌──────────┐     ┌──────────┐
              │aggregator│    │index_server│   │index_writer│
              └──────────┘    └──────────┘     └──────────┘
                    │                │
                    │                ▼
                    │          ┌──────────┐
                    └─────────►│   cli    │
                               └──────────┘
```

## Entry Points

| Entry Point | Binary | Purpose |
|-------------|--------|---------|
| `rs/aggregator/src/main.rs` | `aggregator` | Distributed query aggregation server (sharding) |
| `rs/index_server/src/main.rs` | `index_server` | Main gRPC server for vector operations |
| `rs/index_writer/src/main.rs` | `index_writer` | Offline index building from HDF5/input files |
| `rs/cli/src/main.rs` | `client` | CLI client for interacting with index_server |
| `rs/cli/src/index_viewer.rs` | `index_viewer` | Tool for inspecting index files |
| `rs/demo/src/main.rs` | `demo` | Demo/test binaries for insert/search |

## Module Organization

### index Crate (Core Library)

**Purpose:** Core index implementations and collection management

| Module | Contents |
|--------|----------|
| `collection/` | `Collection` struct, versioning, snapshots, WAL integration |
| `segment/` | Segment trait, mutable/immutable/pending segment implementations |
| `spann/` | Single-user SPANN index (HNSW + IVF) |
| `multi_spann/` | Multi-user SPANN with per-user lazy loading |
| `hnsw/` | HNSW index: `BlockBasedHnsw`, builder, reader, writer |
| `ivf/` | IVF index: `BlockBasedIvf`, posting lists, invalidated IDs |
| `query/` | Query planners, async iterators (AND, OR, term, IDs) |
| `wal/` | Write-ahead log with group commit |
| `optimizers/` | Segment optimization: merge, vacuum |
| `multi_terms/` | Hybrid search combining text terms and vectors |
| `terms/` | Term index for text search |
| `tokenizer/` | Text tokenization (whitespace, stemming) |

### utils Crate (Shared Utilities)

**Purpose:** Low-level utilities used across all crates

| Module | Contents |
|--------|----------|
| `file_io/env.rs` | `Env` trait, `DefaultEnv`, file type abstractions |
| `block_cache/` | `BlockCache` for async I/O caching |
| `distance/` | L2 distance, dot product, SIMD implementations |
| `bloom_filter/` | Blocked bloom filters, builders, readers |
| `kmeans_builder/` | K-means clustering for centroid computation |
| `on_disk_ordered_map/` | On-disk hash tables via `odht` |

### proto Crate

**Purpose:** Compiled gRPC protobuf definitions

- Generated from `.proto` files in repo root
- Used by `index_server`, `aggregator`, `cli`

### config Crate

**Purpose:** Configuration types shared across crates

| Type | Purpose |
|------|---------|
| `CollectionConfig` | Index parameters, quantization, WAL settings |
| `SearchParams` | Top-K, num_probes, ef_construction |
| `AttributeSchema` | Document attribute types |

### quantization Crate

**Purpose:** Vector quantization implementations

| Type | Purpose |
|------|---------|
| `Quantizer` trait | Core quantization interface |
| `NoQuantizer` | Pass-through quantization (f32) |
| `ProductQuantizer` | PQ for vector compression |
| `RabitQBuilder` | RabitQ quantization |

### compression Crate

**Purpose:** Integer sequence compression for posting lists

| Type | Purpose |
|------|---------|
| `EliasFano` | Elias-Fano encoding for sorted integers |
| `NOC` | No-compression fallback |
| `AsyncIntSeqDecoder` | Async iterator for compressed sequences |

### index_server Crate

**Purpose:** gRPC server exposing vector operations

| Module | Contents |
|--------|----------|
| `index_server.rs` | `IndexServerImpl` with create_collection, insert, search, remove, flush |
| `collection_manager.rs` | Manages multiple collections lifecycle |
| `collection_catalog.rs` | Collection metadata registry |
| `http_server.rs` | HTTP admin endpoints |
| `admin_server.rs` | Admin gRPC services |

### aggregator Crate

**Purpose:** Distributed query aggregation

| Module | Contents |
|--------|----------|
| `aggregator.rs` | Fan-out search to shard nodes, aggregate results |
| `shard_manager.rs` | Shard routing logic |
| `node_manager.rs` | Node discovery and health |

## Key File Locations

**Core Data Structures:**
- `rs/index/src/collection/core.rs` - Main Collection struct and lifecycle
- `rs/index/src/segment/mod.rs` - Segment trait and BoxedImmutableSegment enum
- `rs/index/src/spann/index.rs` - SPANN index combining HNSW + IVF
- `rs/index/src/multi_spann/index.rs` - Multi-user SPANN with lazy loading

**Storage Layer:**
- `rs/utils/src/file_io/env.rs` - Env trait for pluggable storage
- `rs/utils/src/block_cache/cache.rs` - BlockCache implementation
- `rs/index/src/vector/async_storage.rs` - Async vector storage

**Index Implementations:**
- `rs/index/src/hnsw/block_based/index.rs` - Block-based HNSW
- `rs/index/src/ivf/block_based/index.rs` - Block-based IVF
- `rs/index/src/wal/wal.rs` - Write-Ahead Log

**Query Execution:**
- `rs/index/src/query/planner.rs` - Query planning
- `rs/index/src/query/async_planner.rs` - Async query execution
- `rs/index/src/query/async_iters/` - Async query iterators

**Server Components:**
- `rs/index_server/src/index_server.rs` - Main gRPC server
- `rs/aggregator/src/aggregator.rs` - Distributed aggregation

**Configuration:**
- `rs/config/src/lib.rs` - CollectionConfig, SearchParams
- `rs/index/src/collection/collection_config.rs` (if exists, else in config)

---

*Structure analysis: 2026-04-04*
