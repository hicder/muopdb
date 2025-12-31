MuopDB - A vector database for AI memories
---

## Introduction
MuopDB is a vector database for machine learning. Currently, it supports:
* Hybrid search: Text search (with stemming support), vector search with filtering.
* Index type: HNSW, IVF, SPANN, Multi-user SPANN. All on-disk.
* Different I/O model: mmap, async I/O (with optional `io_uring` support on Linux).
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
# Start server locally. This is recommended for Mac.
cd target/release
RUST_LOG=info ./index_server --node-id 0 --index-config-path /mnt/muopdb/indices --index-data-path /mnt/muopdb/data --port 9002

# Start server with Docker. Only use this option on Linux.
docker-compose up --build
```
* Now you have an up and running MuopDB `index_server`.
  * You can send gRPC requests to this server (possibly with [Postman](https://www.postman.com/)).
  * You can use Server Reflection in Postman - it will automatically detect the RPCs for MuopDB.
### Examples using Postman
1. Create collection
<img width="802" alt="Screenshot 2025-03-26 at 8 32 05 PM" src="https://github.com/user-attachments/assets/52af33b0-3698-4770-90af-ff679c42ffd6" />


```
{
    "collection_name": "test-collection-2",
    "num_features": 10,
    "wal_file_size": 1024000000,
    "max_time_to_flush_ms": 5000,
    "max_pending_ops": 10,
    "attribute_schema": {
        "attributes": [
            {
                "name": "title",
                "type": "ATTRIBUTE_TYPE_TEXT",
                "language": "english"
            },
            {
                "name": "content",
                "type": "ATTRIBUTE_TYPE_TEXT",
                "language": "english"
            }
        ]
    }
}
```

2. Insert some data

<img width="782" alt="Screenshot 2025-03-26 at 8 24 52 PM" src="https://github.com/user-attachments/assets/6d6bed7d-637d-48c6-96b2-6a512c2f848a" />

```
{
    "collection_name": "test-collection-2",
    "doc_ids": [
        {
            "uuid": "00000000-0000-0000-0000-000000000064"
        }
    ],
    "user_ids": [
        {
            "uuid": "00000000-0000-0000-0000-000000000000"
        }
    ],
    "vectors": [
        100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0
    ],
    "attributes": {
        "values": [
            {
                "value": {
                    "title": {
                        "text_value": "Example Document"
                    },
                    "content": {
                        "text_value": "This is an example document for search demonstration"
                    }
                }
            }
        ]
    }
}
```

3. Search
<img width="603" alt="Screenshot 2025-03-26 at 8 25 40 PM" src="https://github.com/user-attachments/assets/e01cfa34-ade0-467c-b4b5-5d9dbec65e88" />

```
{
    "collection_name": "test-collection-2",
    "params": {
        "ef_construction": 200,
        "record_metrics": false,
        "top_k": 1
    },
    "user_ids": [
        {
            "uuid": "00000000-0000-0000-0000-000000000000"
        }
    ],
    "vector": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0]
}
```

4. Remove

<img width="603" alt="Screenshot 2025-03-26 at 8 25 57 PM" src="https://github.com/user-attachments/assets/7007eb6a-ca96-423d-b866-6ead2c5cbb22" />


```
{
    "collection_name": "test-collection-2",
    "doc_ids": [
        {
            "uuid": "00000000-0000-0000-0000-000000000064"
        }
    ],
    "user_ids": [
        {
            "uuid": "00000000-0000-0000-0000-000000000000"
        }
    ]
}
```

5. Search again
You should see something else
<img width="603" alt="Screenshot 2025-03-26 at 8 26 15 PM" src="https://github.com/user-attachments/assets/33ab4e14-785c-4bd9-a9a0-668cc4c554c0" />

```
{
    "collection_name": "test-collection-2",
    "params": {
        "ef_construction": 200,
        "record_metrics": false,
        "top_k": 1
    },
    "user_ids": [
        {
            "uuid": "00000000-0000-0000-0000-000000000000"
        }
    ],
    "vector": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0]
}
```

This time it should give you something else

6. TermSearch only
<img width="603" alt="Screenshot 2025-03-26 at 8 26 30 PM" src="https://github.com/user-attachments/assets/33ab4e14-785c-4bd9-a9a0-668cc4c554c0" />

```
{
    "collection_name": "test-collection-2",
    "user_ids": [
        {
            "uuid": "00000000-0000-0000-0000-000000000000"
        }
    ],
    "limit": 10,
    "filter": {
        "contains": {
            "path": "content",
            "value": "search"
        }
    }
}
```

This performs a text-only search without requiring a vector, returning documents where the `content` field contains the term "search". You can also search the `title` field or combine multiple filters using `and`/`or` operators.

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
### Phase 3 (Done)
- [x] Features
  - [x] Delete vector from collection
- [x] Database Management
  - [x] Segment optimizer framework
  - [x] Write-ahead-log
  - [x] Segments merger
  - [x] Segments vacuum
### Phase 4 (Done)
- [x] Features
  - [x] Hybrid search
  - [x] Term search only (without vector)
- [x] Database Management
  - [x] Optimizing deletion with bloom filter
  - [x] Optimizing WAL write with thread-safe write group
  - [x] Automatic segment optimizer
  - [x] Non-mmap implementation of SPANN and Term index (with `io_uring` support)

### Building

- Install prerequisites:
  - Rust: https://www.rust-lang.org/tools/install
  - Make sure you're on nightly: `rustup toolchain install nightly`
  - Libraries
```bash
# MacOS (using Homebrew)
brew install protobuf openblas

# Linux (Arch-based)
# On Arch Linux (and its derivatives, such as EndeavourOS, CachyOS):
sudo pacman -Syu protobuf openblas

# Linux (Debian-based)
sudo apt-get install libprotobuf-dev libopenblas-dev
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
Main contributors:
* [Hieu Pham](https://www.linkedin.com/in/phamduchieu/)
* [Son Tuan Vu](https://www.linkedin.com/in/sontuanvu/)
 
This project is done with [TechCare Coaching](https://techcarecoaching.com/). I am mentoring mentees who made contributions to this project.
