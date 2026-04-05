# Coding Conventions

**Analysis Date:** 2025-04-04

## Style Guidelines

**Formatting:**
- Uses `rustfmt` with configuration in `rustfmt.toml`:
  - `reorder_imports = true` - Import statements are sorted
  - `imports_granularity = "Module"` - Imports grouped by module
  - `group_imports = "StdExternalCrate"` - Stdlib imports grouped separately from external crates
- CI enforces formatting via `cargo fmt -- --check`

**Import Organization:**
```rust
// Standard library imports first
use std::cmp::min;
use std::collections::{BinaryHeap, HashMap};

// External crate imports
use anyhow::{anyhow, Context, Result};
use bit_vec::BitVec;
use ordered_float::NotNan;
use quantization::quantization::Quantizer;
use rand::Rng;
use tracing::debug;

// Internal module imports
use crate::hnsw::utils::{BuilderContext, GraphTraversal};
use crate::utils::PointAndDistance;
```

**Naming Conventions:**
- **Files:** `snake_case.rs` (e.g., `core.rs`, `graph_storage.rs`)
- **Directories:** `snake_case` (e.g., `block_based/`, `multi_spann/`)
- **Structs/Enums:** `CamelCase` (e.g., `HnswBuilder`, `WalOpType`)
- **Functions/Methods:** `snake_case` (e.g., `reindex()`, `get_doc_id()`)
- **Variables:** `snake_case` (e.g., `max_neighbors`, `base_directory`)
- **Constants:** `SCREAMING_SNAKE_CASE` or `UPPERCASE` (e.g., `NUM_GROUPS`)
- **Type Parameters:** Single uppercase letter (e.g., `Q: Quantizer`)
- **Generic associated types:** `QuantizedT` pattern

## Error Handling

**Primary Pattern:**
- Uses `anyhow` crate for error handling throughout
- Functions return `anyhow::Result<T>` instead of `std::result::Result<T, E>`
- Error propagation with `?` operator
- Context attachment with `.context()` method

**Error Creation:**
```rust
use anyhow::{anyhow, Context, Result};

// Function signature
fn reindex(&mut self, id_mapping: &[i32]) -> Result<()> {
    let new_point_id = id_mapping.get(point_id as usize)
        .ok_or(anyhow!("point_id {} is larger than size of mapping", point_id))?;
    // ...
}

// With context
let reader = CollectionReader::new(name, base_directory, Some(env));
let collection = rt.block_on(reader.read::<NoQuantizerL2>())
    .context("Failed to read collection")?;
```

## Async Patterns

**Async Runtime:**
- Uses `tokio` with features: `macros`, `rt-multi-thread`, `sync`, `signal`
- Tests use `#[tokio::test]` attribute

**Concurrency Primitives:**
- **Use `async-lock` (not std sync):** For async code, always use `async_lock::RwLock`
  - `rs/index/src/collection/core.rs`: `use async_lock::{RwLock, RwLockUpgradableReadGuard}`
  - `rs/index/src/multi_spann/index.rs`: `use async_lock::RwLock`
- **Use `tokio::sync`:** For channels and oneshot: `tokio::sync::mpsc`, `tokio::sync::oneshot`, `tokio::sync::watch`
- **Use `std::sync`:** For `Arc` only (never for `Mutex`/`RwLock` in async contexts)

**Async I/O:**
- Uses `tokio::fs` for file operations in async contexts
- Uses `tokio::io::{AsyncReadExt, AsyncWriteExt, AsyncSeekExt}`

**Example Async Function:**
```rust
pub async fn get_doc_id(&self, point_id: u32) -> Option<u128> {
    self.posting_lists.get_doc_id(point_id).await.ok()
}
```

## Serialization

**Primary Serialization:**
- Uses `rkyv` for zero-copy deserialization
- Uses `rkyv::util::AlignedVec` for aligned byte storage
- Found in: `rs/index/src/wal/file.rs`, `rs/index/src/wal/entry.rs`

**Configuration Serialization:**
- Uses `serde` with `serde_json` and `serde_yaml` for config files
- `#[derive(Serialize, Deserialize, Debug, Clone)]` pattern

**Example:**
```rust
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TableOfContent {
    pub toc: Vec<String>,
    #[serde(default)]
    pub pending: HashMap<String, Vec<String>>,
    #[serde(default = "default_sequence_number")]
    pub sequence_number: i64,
}
```

## Documentation

**Doc Comment Style:**
- Standard Rust doc comments with `///`
- Comprehensive documentation for public APIs
- `# Arguments`, `# Returns`, `# Warning` sections

**Example:**
```rust
/// Returns the document ID associated with a point ID.
///
/// # Arguments
/// * `point_id` - The internal point ID.
///
/// # Returns
/// * `Option<u128>` - The 128-bit document ID if found, or `None`.
pub async fn get_doc_id(&self, point_id: u32) -> Option<u128> {
    self.posting_lists.get_doc_id(point_id).await.ok()
}

/// Returns the point ID associated with a document ID.
///
/// # Warning
/// This is very expensive and should only be used for testing as it performs a scan.
#[cfg(test)]
pub async fn get_point_id(&self, doc_id: u128) -> Option<u32> {
    // ...
}
```

**README Presence:**
- No per-crate README files found
- Documentation centralized in `AGENTS.md` at repository root

## Specialized Patterns

**Nightly Rust Required:**
- Channel: `nightly` (specified in `rust-toolchain.toml`)
- Required features:
  - `#![feature(auto_traits)]` - in `rs/index/src/lib.rs`
  - `#![feature(min_specialization)]` - in `rs/index/src/lib.rs`
  - `#![feature(portable_simd)]` - in `rs/utils/src/lib.rs`
  - `#![feature(btree_cursors)]` - in `rs/utils/src/lib.rs`
  - `#![feature(new_range_api)]` - in `rs/utils/src/lib.rs`

**Self-Referential Structs:**
- Uses `ouroboros` crate for self-referential struct handling

**Concurrent Data Structures:**
- Uses `dashmap` for concurrent hash maps
- Uses `DashMap<u128, Arc<Spann<Q>>>` pattern for per-user indices

**Logging:**
- Uses `tracing` crate (not `log`)
- `tracing::debug!()` for debug logging
- `tracing-subscriber` with OpenTelemetry support

**Linting:**
- Uses `cargo clippy --all-targets --all-features -- -D warnings`
- Workspace-level lint configuration in `Cargo.toml`:
  - `uninlined_format_args = "allow"`

---

*Convention analysis: 2025-04-04*
