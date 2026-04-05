# Technology Stack

**Analysis Date:** 2026-04-04

## Runtime & Async

**Async Runtime:**
- `tokio` v1.24 with multi-threaded runtime, sync, signal, and macro features
- `futures` v0.3.31 for async utilities
- `async-trait` v0.1 for async trait definitions
- `async-lock` v3.4 for async synchronization primitives

**Concurrency Primitives:**
- `dashmap` v6.1.0 - Concurrent hash map for multi-user indices
- `parking_lot` v0.12.3 - Fast mutexes (with `send_guard` feature)
- `crossbeam` v0.8 - Lock-free data structures and channels
- `rayon` v1.10.0 - Data parallelism for CPU-intensive operations

**Self-Referential Structs:**
- `ouroboros` v0.18.5 - Used extensively for self-referential types in indices
- `atomic_refcell` v0.1.13 - Atomic cell types

## Serialization & RPC

**Serialization:**
- `rkyv` v0.8.10 - Zero-copy deserialization for index data
- `serde` v1.0 + `serde_json` + `serde_yaml` - General serialization
- `prost` v0.13.4 - Protocol Buffers code generation
- `byteorder` v1.5.0 - Byte order handling for packed data

**RPC Framework:**
- `tonic` v0.12.3 - gRPC client/server framework
- `tonic-build` v0.12.3 - Build-time code generation
- `tonic-reflection` v0.12.3 - gRPC server reflection support
- `tokio-stream` v0.1 - Streaming for gRPC

## Storage & I/O

**File I/O Approach:**
- **Async block-based I/O** via custom `Env` trait in `rs/utils/src/file_io/env.rs`
- `memmap2` v0.9.5 - Memory-mapped file support (legacy path)
- `moka` v0.12 - Concurrent cache with futures support
- `nix` v0.29 with `fs` features - Unix-specific file operations

**Platform-Specific I/O:**
- **Linux-only:** `io-uring` v0.6 + `tokio-uring` v0.5 for high-performance async I/O
- **Standard:** `CachedStandard` - Cross-platform async file I/O with block cache

**Object Storage:**
- `object_store` v0.13.1 with AWS features - S3-compatible storage backend

## Vector Processing

**Linear Algebra:**
- `ndarray` v0.16.1 - N-dimensional arrays
- `ndarray-linalg` v0.17.0 with `openblas-system` - Linear algebra operations
- `ndarray-rand` v0.15.0 - Random array generation

**Clustering:**
- `kmeans` v0.11.0 - K-means clustering for vector quantization

**Math Primitives:**
- `num-traits` v0.2.19 - Numeric trait abstractions
- `ordered-float` v4.3.0 - Ordered floating-point types

## Quantization & Compression

**Quantization:**
- `quantization` crate (internal) - Product Quantization (PQ) and NoQuantizer implementations

**Compression:**
- `compression` crate (internal) - Elias-Fano encoding, NOC (no compression)
- `integer-encoding` v4.0.2 - Variable-length integer encoding
- `bitvec` v1.0 - Bit vector operations
- `bit-vec` v0.8.0 - Alternative bit vector implementation
- `roaring` v0.10.6 - Compressed bitmaps for posting lists

## Observability

**Logging:**
- `log` v0.4.22 - Standard logging facade
- `env_logger` v0.11.5 - Environment-based logger
- `tracing` v0.1 - Structured logging and instrumentation
- `tracing-subscriber` v0.3 with `env-filter` and `json` features

**Metrics:**
- `opentelemetry` v0.27 + `opentelemetry_sdk` with tokio runtime
- `opentelemetry-otlp` v0.27 with tonic - OTLP exporter
- `tracing-opentelemetry` v0.28 - Tracing integration
- `prometheus-client` v0.23.1 - Prometheus metrics

**Profiling:**
- `pprof` v0.15 with `prost-codec` features - CPU profiling

## Data Structures

**Collections:**
- `dashmap` v6.1.0 - Concurrent hash map (used for multi-user indices)
- `sorted-vec` v0.8.5 - Sorted vector implementation
- `odht` v0.3.1 - Hash table implementation
- `xxhash-rust` v0.8.15 with `xxh3` - Fast hashing

**String Processing:**
- `rust-stemmers` v1.2 - Text stemming for full-text search

**Randomness:**
- `rand` v0.8.5 - Random number generation

## Build System

**Build Tools:**
- Cargo workspace with 14+ crates
- Rust Edition 2021 (most crates), Edition 2024 (metrics crate)
- Resolver version 2
- Nightly Rust required (features: `auto_traits`, `min_specialization`, `portable_simd`)

**Required System Dependencies:**
- `protobuf` (protoc compiler) - For gRPC code generation
- `openblas` - Linear algebra library for ndarray-linalg

**Development Tools:**
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

---

*Stack analysis: 2026-04-04*
