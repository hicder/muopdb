# Testing Strategy

**Analysis Date:** 2025-04-04

## Test Organization

**Test Location:**
- **Co-located with source:** Tests are inline using `#[cfg(test)] mod tests` at the end of source files
- **89 test modules** found across the codebase
- **No separate `tests/` directories** - all tests embedded in source files

**Test Naming:**
- Test functions: `test_<description>` pattern (e.g., `test_spann_search`, `test_collection`)
- Async tests: `#[tokio::test]` attribute

## Test Types

| Type | Location | Purpose |
|------|----------|---------|
| **Unit Tests** | Inline in source files (`#[cfg(test)]`) | Test individual components in isolation |
| **Integration Tests** | Inline with external I/O | Test component interactions with temp directories |
| **Async Tests** | Same files with `#[tokio::test]` | Test async code paths |
| **Benchmarks** | `rs/benchmarks/src/` | Performance testing with Criterion |

## Test Structure

**Standard Test Module Pattern:**
```rust
#[cfg(test)]
mod tests {
    use super::*;
    // Additional imports for testing
    use std::sync::Arc;
    use tempdir::TempDir;

    #[tokio::test]
    async fn test_feature() {
        let temp_dir = TempDir::new("test_name")
            .expect("Failed to create temporary directory");
        let base_dir = temp_dir.path().to_str().unwrap().to_string();
        
        // Test implementation
        assert_eq!(result, expected);
    }
}
```

**Test-Only Public Methods:**
```rust
/// Returns a reference to the centroids index.
#[cfg(test)]
pub fn get_centroids(&self) -> &BlockBasedHnsw<NoQuantizer<L2DistanceCalculator>> {
    &self.centroids
}
```

## Running Tests

**All Tests:**
```bash
cargo test --release
```

**Specific Package:**
```bash
cargo test --release -p index
```

**Specific Test:**
```bash
cargo test --release -p index -- test_name
```

**With Linting:**
```bash
cargo clippy --all-targets --all-features -- -D warnings
```

## Test Utilities

**Location:** `rs/utils/src/test_utils.rs`

**Available Helpers:**
```rust
// Generate random test vectors
pub fn generate_random_vector(dimension: usize) -> Vec<f32>

// Generic random vector generation
pub fn generate_random_vector_generic<T>(dimension: usize) -> Vec<T>
where
    Standard: Distribution<T>
```

**Usage in Tests:**
```rust
use utils::test_utils::generate_random_vector;

let vectors = (0..num_vectors)
    .map(|_| generate_random_vector(num_features))
    .collect::<Vec<_>>();
```

**Temp Directory Pattern:**
- Uses `tempdir` crate for temporary test directories
- Pattern: `TempDir::new("test_description").expect("Failed to create temporary directory")`

## Benchmarks

**Location:** `rs/benchmarks/src/`

| Benchmark | File | Purpose |
|-----------|------|---------|
| **Insertion** | `insertion.rs` | Vector insertion performance |
| **WAL Insertion** | `wal_insertion.rs` | Write-ahead log performance |
| **Vacuum** | `vacuum.rs` | Vacuum optimizer performance |
| **Deletion & Vacuum** | `deletion_and_vacuum.rs` | Combined deletion/vacuum performance |

**Running Benchmarks:**
```bash
cargo bench -p benchmarks
```

**Benchmark Structure:**
```rust
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

fn bench_insertion(c: &mut Criterion) {
    let mut group = c.benchmark_group("Insertion");
    group.sample_size(10);
    
    group.bench_with_input(
        BenchmarkId::new("Insertion", 10000),
        &10000,
        |bencher, _| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            bencher.iter_with_setup(
                || { /* setup */ },
                |input| { /* benchmark code */ }
            );
        },
    );
    
    group.finish();
}

criterion_group!(benches, bench_insertion);
criterion_main!(benches);
```

## CI/CD

**Configuration:** `.github/workflows/rust.yml`

**Jobs:**
1. **Build & Test (Ubuntu):**
   - Runs in Docker via `docker build .`
   - Executes full test suite

2. **Check Formatting:**
   - Uses stable Rust toolchain
   - Runs `cargo fmt -- --check`

**Triggers:**
- Push to `master` branch
- Pull requests to `master` branch

## Test Examples

**Async Test with Tokio:**
```rust
// From rs/index/src/spann/index.rs
#[tokio::test]
async fn test_spann_search() {
    let temp_dir = tempdir::TempDir::new("spann_search_test")
        .expect("Failed to create temporary directory");
    let base_dir = temp_dir.path().to_str()
        .expect("Failed to convert temporary directory path to string")
        .to_string();

    // Build and test...
    let results = spann.search(query, &params, None).await
        .expect("IVF search should return a result");
    
    assert_eq!(results.id_with_scores.len(), k);
    assert_eq!(results.id_with_scores[0].doc_id, 4);
}
```

**Multi-threaded WAL Test:**
```rust
// From rs/index/src/collection/core.rs
#[tokio::test]
async fn test_collection_wal_multi_threaded() -> Result<()> {
    const USER_ID: u128 = 0;
    const NUM_GROUPS: usize = 10;
    const OPERATIONS_PER_GROUP: usize = 4;
    
    // Spawn multiple tasks writing to WAL concurrently
    let mut join_set = JoinSet::new();
    for group_id in 0..NUM_GROUPS {
        join_set.spawn(async move {
            // Perform operations...
        });
    }
    
    while let Some(result) = join_set.join_next().await {
        result??;
    }
    
    Ok(())
}
```

## Key Test Patterns

**Environment Setup Helper:**
```rust
fn create_env() -> Arc<Box<dyn Env>> {
    let config = EnvConfig {
        file_type: FileType::CachedStandard,
        ..EnvConfig::default()
    };
    Arc::new(Box::new(DefaultEnv::new(config)))
}
```

**Collection Test Setup:**
```rust
let collection_name = "test_collection";
let temp_dir = TempDir::new(collection_name)?;
let base_directory: String = temp_dir.path().to_str().unwrap().to_string();
let segment_config = CollectionConfig::default_test_config();
let collection = Arc::new(
    Collection::<NoQuantizerL2>::new(
        collection_name.to_string(),
        base_directory.clone(),
        segment_config,
        None,
    )
    .await
    .unwrap(),
);
```

---

*Testing analysis: 2025-04-04*
