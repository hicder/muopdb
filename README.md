MuopDB - A vector database for AI memories
---

## Introduction
MuopDB is a vector database for machine learning. Currently, it supports:
* Index type: HNSW, IVF, SPANN, Multi-user SPANN. All on-disk with mmap.
* Quantization: product quantization

## Why MuopDB?
MuopDB supports multiple users by default. What that means is, each user will have its own vector index, within the same collection. The use-case for this is to build memory for LLMs.
Think of it as:
* Each user will have its own memory
* Each user can still search a shared knowledge base.

All users' indices will be stored in a few files, reducing operational complexity.

## Quick Start

* Build MuopDB. Refer to this [instruction](https://github.com/hicder/muopdb?tab=readme-ov-file#building).
* Prepare necessary `data` and `indices` directories. On Mac, you might want to change these directories since root directory is read-only, i.e: `~/mnt/muopdb/`.
```
mkdir -p /mnt/muopdb/indices
mkdir -p /mnt/muopdb/data
```
* Start MuopDB `index_server` with the directories we just prepared using one of these methods:
```bash
# Start server locally
cd target/release
RUST_LOG=info ./index_server --node-id 0 --index-config-path /mnt/muopdb/indices --index-data-path /mnt/muopdb/data --port 9002

# Start server with Docker
docker-compose up --build
```
* Now you have an up and running MuopDB `index_server`.
  * You can send gRPC requests to this server (possibly with [Postman](https://www.postman.com/)).
  * Use [muopdb.proto](https://github.com/hicder/muopdb/blob/b2b3c4bf84d900e118341bd95eae0e32ce65d3c0/rs/proto/proto/muopdb.proto#L38-L48) for Service Definition. Refer to [this guide](https://learning.postman.com/docs/sending-requests/grpc/using-service-definition/) for more information.

### Examples using Postman
1. Create collection
<img width="603" alt="Screenshot 2025-01-16 at 11 14 23 AM" src="https://github.com/user-attachments/assets/cadf00c4-199f-4756-8446-7fb08de2b0c0" />

```
{
    "collection_name": "test-collection-2",
    "num_features": 10
}
```

2. Insert some data
<img width="603" alt="Screenshot 2025-01-17 at 9 45 20 AM" src="https://github.com/user-attachments/assets/ec15a3b7-3a0a-44a3-a929-29ac9b7a47fc" />

```
{
    "collection_name": "test-collection-12",
    "high_ids": [
        0
    ],
    "low_ids": [
        4
    ],
    "high_user_ids": [
        0
    ],
    "low_user_ids": [
        0
    ],
    "vectors": [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0
    ]
}
```


3. Flush
<img width="603" alt="Screenshot 2025-01-16 at 10 51 42 AM" src="https://github.com/user-attachments/assets/83f0d12c-afde-47f5-9238-eedf31a4dad5" />

```
{
    "collection_name": "test-collection-2",
}
```

4. Query
<img width="603" alt="Screenshot 2025-01-17 at 9 45 31 AM" src="https://github.com/user-attachments/assets/5859453b-5423-4321-a032-337a0a061ac1" />

```
{
    "collection_name": "test-collection-12",
    "ef_construction": 100,
    "record_metrics": false,
    "top_k": 1,
    "high_user_ids": [0],
    "low_user_ids": [0],
    "vector": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0, 9.0, 9.0]
}
```

## Plans
### Phase 0 (Done)
- [x] Query path
  - [x] Vector similarity search
  - [x] Hierarchical Navigable Small Worlds (HNSW)
  - [x] Product Quantization (PQ)
- [x] Indexing path
  - [x] Support periodic offline indexing
- [x] Database Management
  - [x] Doc-sharding & query fan-out with aggregator-leaf architecture
  - [x] In-memory & disk-based storage with mmap
### Phase 1 (Done)
- [x] Query & Indexing
  - [x] Inverted File (IVF)
  - [x] Improve locality for HNSW
  - [x] SPANN
### Phase 2 (Done)
- [x] Query
  - [x] Multiple index segments
  - [x] L2 distance
- [x] Index
  - [x] Optimizing index build time
  - [x] Elias-Fano encoding for IVF
  - [x] Multi-user SPANN index
### Phase 3 (Ongoing)
- [ ] Features
  - [ ] Delete vector from collection
  - [ ] Querying mutable segment
  - [ ] [RabitQ quantization](https://arxiv.org/abs/2405.12497)
  - [ ] Embedded MuopDB (with Python binding)
  - [ ] Cloud MuopDB
- [ ] Database Management
  - [ ] Segment optimizers (vacumn, merge)

### Building

- Install prerequisites:
  - Rust: https://www.rust-lang.org/tools/install
  - Libraries
```bash
# MacOS (using Homebrew)
brew install hdf5 protobuf openblas

# Linux (Arch-based)
# On Arch Linux (and its derivatives, such as EndeavourOS, CachyOS):
sudo pacman -Syu hdf5 protobuf openblas

# Linux (Debian-based)
sudo apt-get install libhdf5-dev libprotobuf-dev libopenblas-dev
```

- Build from Source:
```bash
git clone https://github.com/hicder/muopdb.git
cd muopdb

# Build
cargo build --release

# Run tests
cargo test --release
```

## Contributions
This project is done with [TechCare Coaching](https://techcarecoaching.com/). I am mentoring mentees who made contributions to this project.
