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
* Prepare necessary `data` and `indices` directories
```
mkdir -p /mnt/muopdb/indices
mkdir -p /mnt/muopdb/data
```
* Run MuopDB `index_server`. (Refer to build instruction down below)
```
./target/release/index_server --node-id 0 --index-config-path /mnt/muopdb/indices --index-data-path /mnt/muopdb/data --port 9002
```
* Now you have an up and running MuopDB `index_server`.
  * You can send gRPC requests to this server (possibly with [Postman](https://www.postman.com/)).
  * Refer to [muopdb.proto](https://github.com/hicder/muopdb/blob/master/rs/proto/proto/muopdb.proto) for the APIs
### Examples using Postman
1. Create collection
<img width="603" alt="Screenshot 2025-01-16 at 10 51 23 AM" src="https://github.com/user-attachments/assets/881b8925-d6ad-4174-966a-8f20742982b8" />

2. Insert some data
<img width="603" alt="Screenshot 2025-01-16 at 10 51 32 AM" src="https://github.com/user-attachments/assets/8dfb622c-31ca-44c1-b174-8ca678a5c32c" />

3. Flush
<img width="603" alt="Screenshot 2025-01-16 at 10 51 42 AM" src="https://github.com/user-attachments/assets/83f0d12c-afde-47f5-9238-eedf31a4dad5" />

4. Query
<img width="603" alt="Screenshot 2025-01-16 at 10 51 52 AM" src="https://github.com/user-attachments/assets/bc564d91-8bc3-47bc-ada7-89085831cdce" />

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
# from top-level workspace
cargo build --release
```
Test:
```
cargo test --release
```
## Contributions
This project is done with [TechCare Coaching](https://techcarecoaching.com/). I am mentoring mentees who made contributions to this project.
