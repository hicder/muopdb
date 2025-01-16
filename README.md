# MuopDB - A vector database for AI memories

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
* Build MuopDB. Refer to this [instruction](https://github.com/hicder/muopdb?tab=readme-ov-file#building)
* Prepare necessary `data` and `indices` directories. On Mac, you might want to change these directories since root directory is read-only.
```
mkdir -p /mnt/muopdb/indices
mkdir -p /mnt/muopdb/data
```
* Run MuopDB `index_server` with the directories we just prepared.
```
./target/release/index_server --node-id 0 --index-config-path /mnt/muopdb/indices --index-data-path /mnt/muopdb/data --port 9002
```
* Now you have an up and running MuopDB `index_server`.
  * You can send gRPC requests to this server (possibly with [Postman](https://www.postman.com/)).
  * Use [muopdb.proto](https://github.com/hicder/muopdb/blob/master/rs/proto/proto/muopdb.proto) for Service Definition. [Guide](https://learning.postman.com/docs/sending-requests/grpc/using-service-definition/)
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
<img width="603" alt="Screenshot 2025-01-16 at 10 51 32 AM" src="https://github.com/user-attachments/assets/8dfb622c-31ca-44c1-b174-8ca678a5c32c" />
```
{
    "collection_name": "test-collection-2",
    "ids": [4],
    "user_ids": [0],
    "vectors": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
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
<img width="603" alt="Screenshot 2025-01-16 at 11 14 37 AM" src="https://github.com/user-attachments/assets/fc0e0332-37e1-4923-962c-87efce5d7a56" />
```
{
    "collection_name": "test-collection-2",
    "ef_construction": 100,
    "record_metrics": false,
    "top_k": 1,
    "user_ids": [0],
    "vector": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
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
- [ ] Database Management
  - [ ] Segment optimizers (vacumn, merge)

## Building
Install prerequisites:
* Rust: https://www.rust-lang.org/tools/install
* Libraries
```
# MacOS: Use Homebrew
brew install hdf5 protobuf openblas

# Linux: Use your package manager.
# On Arch Linux (and its derivatives, such as EndeavourOS, CachyOS):
sudo pacman -Syu hdf5 protobuf openblas
```
Build:
```
cargo build --release
```
Test:
```
cargo test --release
```
## Contributions
This project is done with [TechCare Coaching](https://techcarecoaching.com/). I am mentoring mentees who made contributions to this project.
