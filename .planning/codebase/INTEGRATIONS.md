# External Integrations

**Analysis Date:** 2026-04-04

## gRPC Services

### IndexServer (`IndexServer`)
- **Primary Service:** Vector database operations
- **Location:** `rs/proto/proto/muopdb.proto`
- **Port:** Configurable (typically specified via CLI/config)
- **Methods:**
  - `CreateCollection` - Create a new vector collection
  - `Search` - Vector similarity search
  - `TermSearch` - Full-text term search
  - `Insert` - Insert vectors with metadata
  - `Remove` - Delete vectors by ID
  - `InsertPacked` - High-throughput binary packed insert
  - `Flush` - Force segment flush to disk

### IndexServerAdmin (`IndexServerAdmin`)
- **Admin Service:** Collection management operations
- **Location:** `rs/proto/proto/admin.proto`
- **Methods:**
  - `GetSegments` - List collection segments with sizes
  - `MergeSegments` - Trigger segment optimization merge

### Aggregator (`Aggregator`)
- **Distributed Query Service:** Fan-out search across shards
- **Location:** `rs/proto/proto/aggregator.proto`
- **Methods:**
  - `Get` - Distributed vector search across multiple index servers
- **Note:** Delegates to underlying `IndexServer` instances

## Protocol Buffers

**Proto Definitions Location:**
- `rs/proto/proto/muopdb.proto` - Core index server API (275 lines)
- `rs/proto/proto/admin.proto` - Admin operations (30 lines)
- `rs/proto/proto/aggregator.proto` - Distributed query API (24 lines)

**Generated Code Location:**
- Built via `rs/proto/build.rs` using `tonic-build`
- Output directory: `$OUT_DIR` (Cargo's build output)
- Generates: `muopdb_descriptor.bin`, `aggregator_descriptor.bin`

**Build Features:**
- Protobuf optional fields enabled (`--experimental_allow_proto3_optional`)
- Both client and server code generated
- Reflection support via `tonic-reflection`

## External Dependencies

### Object Storage (AWS S3)
| Dependency | Type | Purpose |
|------------|------|---------|
| `object_store` v0.13.1 | Crate (with `aws` feature) | S3-compatible object storage backend |
| - | Service | Cold storage, backup, distributed storage |

**Configuration:**
- Endpoint, region, access key, secret key via `ObjectStoreConfig`
- File type: `FileType::ObjectStore` in `EnvConfig`

### OpenTelemetry Backend
| Dependency | Type | Purpose |
|------------|------|---------|
| `opentelemetry-otlp` | Crate | OTLP protocol exporter |
| OTel Collector/Jaeger | Service | Distributed tracing backend (external) |

**Trace Instrumentation:**
- `tracing-opentelemetry` integration
- OTLP over tonic for gRPC transport

### HTTP Client (Demo/Tools)
| Dependency | Type | Purpose |
|------------|------|---------|
| `reqwest` v0.12.11 | Crate | HTTP client for external API calls |
| `hyper` v1.6.0 | Crate | HTTP server for metrics endpoint |

**Usage:**
- `reqwest` used in demo crate for external data fetching
- `hyper` used in `index_server` for Prometheus metrics endpoint

## System Dependencies

### Build-Time Requirements
| Dependency | Required For | Installation |
|------------|--------------|--------------|
| `protoc` (protobuf) | gRPC code generation | macOS: `brew install protobuf` / Arch: `protobuf` / Debian: `libprotobuf-dev` |
| `openblas` | Linear algebra (ndarray-linalg) | macOS: `brew install openblas` / Arch: `openblas` / Debian: `libopenblas-dev` |

### Platform-Specific
| Platform | Feature | Dependencies |
|----------|---------|--------------|
| Linux | io_uring support | Kernel 5.1+ with io_uring enabled |
| All | Block cache | Standard async I/O |

## Data Flow Integration Points

### Insert Path
1. gRPC client → `IndexServer::Insert`
2. `Collection::write_to_wal()` → WAL (group commit)
3. `MutableSegment::insert_for_user()`
4. `MultiSpannBuilder::insert()`

### Search Path
1. gRPC client → `IndexServer::Search`
2. `Collection::get_snapshot()`
3. Parallel segment search
4. `MultiSpannIndex::search_for_user()` → `Spann::search()`
5. HNSW centroids search → IVF posting lists retrieval

### Distributed Search (Aggregator)
1. Client → `Aggregator::Get`
2. Fan-out to multiple `IndexServer` instances
3. Merge results, deduplicate by user/doc ID
4. Return aggregated top-k results

## Environment Configuration

**Critical Configuration Files:**
- `rs/config/src/collection.rs` - Default collection parameters
- `rs/config/src/index_writer.rs` - Index writer configuration

**Required for Object Store:**
- `endpoint` - S3-compatible endpoint URL
- `region` - AWS region or custom
- `access_key_id` - AWS access key
- `secret_access_key` - AWS secret key

**Metrics/Tracing:**
- `OTEL_EXPORTER_OTLP_ENDPOINT` - OpenTelemetry collector URL
- `OTEL_SERVICE_NAME` - Service name for traces

## Service Ports

**Default Ports (configurable via CLI):**
- IndexServer: Configurable via `--index-server-port`
- Aggregator: Configurable via `--port`
- Metrics (Prometheus): Configurable HTTP endpoint

**Binary Entry Points:**
- `rs/index_server/` - Main vector database server
- `rs/aggregator/` - Distributed query aggregator
- `rs/cli/` - Client CLI tool
- `rs/demo/` - Demo utilities (insert, search)
- `rs/index_writer/` - Offline index building tools

---

*Integration audit: 2026-04-04*
