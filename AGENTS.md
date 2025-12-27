# AGENTS.md

Guidelines for AI coding agents working in the MuopDB repository.

## Project Overview

MuopDB is a multi-user vector database for AI memories written in Rust. It uses HNSW, IVF, and SPANN indices with memory-mapped files and a segment-based architecture.

## Build, Test, and Lint Commands

### Prerequisites (macOS)

```bash
brew install hdf5 protobuf openblas
```

### Essential Commands

```bash
# Build (always use --release for performance)
cargo build --release

# Run all tests
cargo test --release

# Run a single test by name
cargo test --release -p index -- test_name

# Run tests in a specific crate
cargo test --release -p index
cargo test --release -p utils
cargo test --release -p quantization

# Run with output visible
cargo test --release -p index -- test_name --nocapture

# Run benchmarks
cargo bench -p benchmarks

# Lint (required to pass in CI)
cargo clippy --all-targets --all-features -- -D warnings

# Format check (required to pass in CI)
cargo fmt -- --check

# Apply formatting
cargo fmt
```

### Toolchain

- **Nightly Rust required** - see `rust-toolchain.toml`
- Uses unstable features: `auto_traits`, `min_specialization`, `portable_simd`

## Code Style Guidelines

### Import Organization (rustfmt.toml)

Imports are automatically organized by `cargo fmt`:
1. Standard library (`std::`)
2. External crates
3. Internal crate modules (`crate::`, `super::`)

```rust
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use dashmap::DashMap;
use log::{debug, info};

use crate::segment::Segment;
use super::snapshot::Snapshot;
```

### Naming Conventions

- **Types/Structs**: `PascalCase` - `MutableSegment`, `SearchResult`
- **Functions/Methods**: `snake_case` - `ann_search`, `write_to_wal`
- **Constants**: `SCREAMING_SNAKE_CASE` - `INTERNAL_METRICS`
- **Generics**: Single uppercase letters with trait bounds - `Q: Quantizer`
- **Test functions**: `test_` prefix - `test_multi_term_builder`

### Type Conventions

- **IDs**: Use `u128` for user IDs and document IDs
- **Indices**: Use `u32` for internal point indices
- **Dimensions**: Use `usize` for vector dimensions and counts
- **Vectors**: Use `&[f32]` for vector data, `Vec<f32>` when owned

### Error Handling

Use `anyhow` for error handling throughout:

```rust
use anyhow::{Ok, Result};

fn do_something() -> Result<()> {
    let file = std::fs::File::open(path)?;
    // ...
    Ok(())
}
```

For gRPC handlers, convert to `tonic::Status`:
```rust
-> Result<tonic::Response<T>, tonic::Status>
```

### Async Patterns

- Use `async_lock` for async mutexes (not `std::sync::Mutex`)
- Use `tokio::sync` for channels and async primitives
- Use `#[tonic::async_trait]` for async trait implementations
- Use `#[async_trait::async_trait]` for general async traits

```rust
use async_lock::RwLock;
use tokio::sync::mpsc;

#[async_trait::async_trait]
pub trait Segment {
    async fn insert(&self, doc_id: u128, data: &[f32]) -> Result<()>;
}
```

### Concurrency Patterns

- Use `DashMap` for concurrent hash maps
- Use `Arc<RwLock<T>>` for shared mutable state
- Use `parking_lot` mutexes for synchronous code
- Use `AtomicRefCell` for interior mutability in builders

### Documentation

Use `///` doc comments for public APIs with sections:

```rust
/// Performs approximate nearest neighbor search.
///
/// # Arguments
/// * `query` - The query vector.
/// * `k` - Number of results to return.
///
/// # Returns
/// * `SearchResult` - The search results.
pub async fn ann_search(&self, query: &[f32], k: usize) -> SearchResult {
```

### Test Structure

Tests go in a `#[cfg(test)]` module at the bottom of the file:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use tempdir::TempDir;

    #[test]
    fn test_feature_name() {
        // Arrange
        let temp_dir = TempDir::new("test").unwrap();
        
        // Act
        let result = do_something();
        
        // Assert
        assert!(result.is_ok());
    }
}
```

### Clippy Allowances

The workspace allows `uninlined_format_args`. Common local allowances:

```rust
#[allow(dead_code)]           // Unused fields needed for ownership
#[allow(clippy::too_many_arguments)]  // Complex constructors
#[allow(clippy::module_inception)]    // mod.rs with same-name submodule
#[allow(clippy::needless_late_init)]  // Conditional initialization
```

## Key Dependencies

| Crate | Purpose |
|-------|---------|
| `anyhow` | Error handling |
| `async-lock` | Async mutexes/RwLocks |
| `dashmap` | Concurrent HashMap |
| `memmap2` | Memory-mapped files |
| `ouroboros` | Self-referential structs |
| `rkyv` | Zero-copy serialization |
| `rayon` | Parallel iterators |
| `tokio` | Async runtime |
| `tonic` | gRPC framework |

## Workspace Structure

All Rust code lives in `rs/`:

| Crate | Description |
|-------|-------------|
| `index/` | Core indices (SPANN, HNSW, IVF), collections, WAL |
| `index_server/` | gRPC server implementation |
| `quantization/` | Vector quantization (PQ, NoQ) |
| `utils/` | Distance functions, bloom filters, caching |
| `config/` | Configuration types |
| `proto/` | Protobuf definitions |
| `aggregator/` | Distributed query routing |

## Common Patterns

### Creating New Index Types

Index types implement the `Quantizer` trait and work with `Spann<Q>`.

### Adding New Tests

Place tests in the same file with `#[cfg(test)]` module. Use `tempdir` for filesystem tests.

### Memory-Mapped Files

Use `memmap2::Mmap` for read-only mmap, store both `File` and `Mmap` to keep file handle alive.
