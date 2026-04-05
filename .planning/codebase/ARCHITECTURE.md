# Architecture Overview

**Analysis Date:** 2026-04-04

## High-Level Design

MuopDB is a multi-user vector database built on a segment-based LSM-tree-like architecture. The core design separates mutable in-memory data from immutable on-disk segments, enabling high-throughput writes while maintaining efficient search performance. The system uses a Write-Ahead Log (WAL) with group commit for durability, and supports multi-tenant workloads through user-isolated indices within shared collections.

The database implements SPANN (Search-Partitioned Approximate Nearest Neighbor) as its primary index type - a two-level index combining HNSW for centroid navigation with IVF for posting list retrieval. The storage layer uses async block-based I/O with an `Env` trait abstraction supporting multiple backends: mmap, cached standard I/O, io_uring (Linux), and object storage.

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

1. **gRPC Request** → `IndexServerImpl::insert()` receives packed insert request
2. **WAL Write** → `Collection::write_to_wal()` writes to WAL with group commit
3. **Channel Send** → Operation sent to internal processing channel
4. **Mutable Segment** → `MutableSegment::insert_for_user()` writes to in-memory SPANN builder
5. **Background Application** → `process_one_op()` applies from channel to mutable segment

### Search Path

1. **gRPC Request** → `IndexServerImpl::search()` receives query vector and user IDs
2. **Snapshot Creation** → `Collection::get_snapshot()` pins current version
3. **Parallel Segment Search** → `SnapshotWithQuantizer::search_for_users()` searches all segments in parallel
4. **Multi-Spann Lookup** → `MultiSpannIndex::search_for_user()` retrieves or creates user index lazily
5. **SPANN Search** → `Spann::search()` → HNSW centroids search → IVF posting lists search
6. **Result Aggregation** → Results merged across segments, sorted by distance

### Flush Path

1. **Trigger** → Auto-flush based on time or pending ops threshold
2. **Segment Swap** → Mutable segment swapped to pending (write lock on `mutable_segments`)
3. **Build Immutable** → `PendingMutableSegment::build()` writes SPANN index to disk
4. **Load Index** → `MultiSpannReader::read()` loads immutable segment
5. **Apply Deletions** → Pending deletions replayed on new immutable segment
6. **Version Update** → `add_segments()` atomically updates Table of Content
7. **WAL Trim** → `trim_wal()` removes entries up to flushed sequence number

## Key Abstractions

### Segment Abstraction

```rust
#[async_trait::async_trait]
pub trait Segment {
    async fn insert(&self, doc_id: u128, data: &[f32]) -> Result<()>>;
    async fn remove(&self, user_id: u128, doc_id: u128) -> Result<bool>;
    async fn may_contain(&self, doc_id: u128) -> bool;
}
```

Three implementations:
- `MutableSegment`: In-memory, writable, uses `MultiSpannBuilder`
- `ImmutableSegment`: Disk-based, read-only, uses `MultiSpannIndex`
- `PendingSegment`: In-progress merge/vacuum operations

### Storage Env Abstraction

```rust
#[async_trait]
pub trait Env: Send + Sync {
    async fn open(&self, path: &str) -> Result<OpenResult>;
    async fn open_append(&self, path: &str) -> Result<OpenAppendResult>;
    async fn close(&self, file_id: FileId) -> Result<()>;
}
```

Supports `FileType::MMap`, `FileType::CachedStandard`, `FileType::CachedIoUring`, `FileType::ObjectStore`

### Quantizer Trait

```rust
pub trait Quantizer: Send + Sync {
    type QuantizedT: VectorT<Self> + Send + Sync;
    fn quantize(&self, value: &[f32]) -> Vec<Self::QuantizedT>;
    fn distance(&self, query: &[Self::QuantizedT], point: &[Self::QuantizedT], ...) -> f32;
}
```

Implementations: `NoQuantizer<L2DistanceCalculator>`, `ProductQuantizer<L2DistanceCalculator>`

### BoxedImmutableSegment

Enum wrapping segment types for type erasure:
- `FinalizedSegment`: Complete immutable segment
- `PendingSegment`: Segment undergoing merge/vacuum
- `MockedNoQuantizationSegment`: Test mock

## Segment Architecture

```
Collection
├── MutableSegment (in-memory, writable)
│   └── MultiSpannBuilder (in-memory index building)
├── PendingMutableSegment (during flush/build)
└── ImmutableSegments (read-only, disk-based)
    ├── FinalizedSegment (SPANN on disk with MultiSpannIndex)
    └── PendingSegment (during merge/vacuum operations)
```

## SPANN Index Structure

```
Spann<Q>
├── centroids: BlockBasedHnsw<NoQuantizer<L2DistanceCalculator>>
│   └── Hierarchical Navigable Small World for centroid navigation
└── posting_lists: BlockBasedIvf<Q>
    └── Inverted File Index with quantized vectors
```

Search: HNSW finds nearest centroids → IVF searches corresponding posting lists → Re-rank by exact distance

## Versioning and Concurrency

- **TableOfContent (TOC)**: Defines active segments per version
- **Atomic Versioning**: TOC files written atomically via rename
- **Snapshot Pinning**: Reference counting prevents segment deletion during search
- **Multi-layered Locking**:
  - `versions_info`: RwLock for version transitions
  - `mutable_segments`: RwLock for segment swapping
  - `all_segments`: DashMap for concurrent immutable access
  - `flushing`: Mutex ensuring single flush operation

## Multi-User Support

`MultiSpannIndex<Q>` uses `DashMap<u128, Arc<Spann<Q>>>` for per-user indices:
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

---

*Architecture analysis: 2026-04-04*
