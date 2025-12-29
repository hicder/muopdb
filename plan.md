# Implementation Plan: MultiTermIndex with FileIO Support

## Problem Statement

Currently, `MultiTermIndex` uses `TermIndex::new()` which creates term indexes without file I/O support. This means `get_posting_list_iterator_block_based()` will always fail when called through `AsyncPlanner`, since it requires the `file_io` field to be set.

## Goal

Enable `MultiTermIndex` to create `TermIndex` instances with file I/O support, allowing async block-based iteration for better I/O performance.

---

## Proposed Changes

### [MODIFY] [index.rs](file:///Users/hieu/code/muopdb/rs/index/src/multi_terms/index.rs)

Add `Env` support to `MultiTermIndex`:

```rust
pub struct MultiTermIndex {
    term_indexes: DashMap<u128, Arc<TermIndex>>,
    user_index_info: TermIndexInfoHashTable,
    base_directory: String,
    // NEW: Optional Env for async file I/O
    env: Option<Arc<dyn Env + Send + Sync>>,
}
```

**New methods:**

1. `new_with_env(base_directory: String, env: Arc<dyn Env + Send + Sync>) -> Result<Self>`
   - Async constructor that stores the env for later use

2. `get_or_create_index_async(&self, user_id: u128) -> Result<Arc<TermIndex>>`
   - Async version that uses `TermIndex::new_with_file_io` when env is available
   - Falls back to sync `TermIndex::new` if no env is set

---

### [MODIFY] [async_planner.rs](file:///Users/hieu/code/muopdb/rs/index/src/query/async_planner.rs)

Update `AsyncPlanner::new` to be async and use the async index creation:

```rust
impl AsyncPlanner {
    pub async fn new(
        user_id: u128,
        query: DocumentFilter,
        multi_term_index: Arc<MultiTermIndex>,
        attribute_schema: Option<AttributeSchema>,
    ) -> Result<Self> {
        // Use async method to get index with file_io support
        let term_index = multi_term_index.get_or_create_index_async(user_id).await?;
        Ok(Self { user_id, query, term_index, attribute_schema })
    }
}
```

---

### [MODIFY] [immutable_segment.rs](file:///Users/hieu/code/muopdb/rs/index/src/segment/immutable_segment.rs)

Update `ImmutableSegment` to accept and pass `Env`:

```rust
pub struct ImmutableSegment<Q: Quantizer> {
    index: MultiSpannIndex<Q>,
    name: String,
    multi_term_index: Option<Arc<MultiTermIndex>>,
}

impl<Q: Quantizer> ImmutableSegment<Q> {
    // NEW: Async constructor with Env support
    pub async fn new_with_env(
        index: MultiSpannIndex<Q>,
        name: String,
        terms_dir: Option<String>,
        env: Option<Arc<dyn Env + Send + Sync>>,
    ) -> Self {
        let multi_term_index = match (terms_dir, env) {
            (Some(dir), Some(env)) => {
                MultiTermIndex::new_with_env(dir, env).await.ok().map(Arc::new)
            }
            (Some(dir), None) => {
                MultiTermIndex::new(dir).ok().map(Arc::new)
            }
            _ => None,
        };
        Self { index, name, multi_term_index }
    }
}
```

Update `search_terms_for_user_async` to use async planner creation:

```rust
pub async fn search_terms_for_user_async(...) -> Vec<u32> {
    // ...
    let planner = AsyncPlanner::new(user_id, filter, multi_term_index, attribute_schema).await?;
    // ...
}
```

---

### Upstream Callers

The following locations create `MultiTermIndex::new()` or `ImmutableSegment::new()` and may need updates:

| File | Line | Usage |
|------|------|-------|
| `segment/immutable_segment.rs` | 28 | `MultiTermIndex::new(dir)` |
| `collection/snapshot.rs` | TBD | Creates ImmutableSegment |
| `query/planner.rs` | 227, 469 | `MultiTermIndex::new(base_directory)` |
| `multi_spann/index.rs` | 876, 966 | `MultiTermIndex::new(term_dir)` |
| `segment/mutable_segment.rs` | 385, 491 | `MultiTermIndex::new(terms_dir)` |
| `multi_terms/writer.rs` | 245, 376, 444, 515 | Test code using `MultiTermIndex::new` |

---

## Implementation Order

1. **Phase 1**: Modify `MultiTermIndex`
   - Add `env` field
   - Add `new_with_env` async constructor
   - Add `get_or_create_index_async` method
   - Keep existing sync methods for backward compatibility

2. **Phase 2**: Update `AsyncPlanner`
   - Make `new` async
   - Update all callers to await the constructor

3. **Phase 3**: Update `ImmutableSegment`
   - Add `new_with_env` async constructor
   - Update callers that need async term index support

4. **Phase 4**: Update upstream callers
   - Propagate `Env` from `Collection`/`Snapshot` level down to segments
   - Update tests to use the new constructors where appropriate

---

## Verification Plan

### Unit Tests
- Add test for `MultiTermIndex::new_with_env` and `get_or_create_index_async`
- Verify `get_posting_list_iterator_block_based` works through `AsyncPlanner`

### Integration Tests
- Test full search flow with async planner using block-based iteration
- Ensure backward compatibility with sync `MultiTermIndex::new`

### Commands
```bash
cargo test -p index multi_term
cargo test -p index async_planner
cargo test -p index immutable_segment
```
