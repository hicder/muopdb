# MuopDB - A vector database for machine learning

## Introduction

MuopDB is a vector database for machine learning. This project is done under [TechCare Coaching](https://techcarecoaching.com/). It plans to support the following features:
### V0 (Done)
- [x] Query path
  - [x] Vector similarity search
  - [x] Hierarchical Navigable Small Worlds (HNSW)
  - [x] Product Quantization (PQ)
- [x] Indexing path
  - [x] Support periodic offline indexing
- [x] Database Management
  - [x] Doc-sharding & query fan-out with aggregator-leaf architecture
  - [x] In-memory & disk-based storage with mmap
### V1
- [ ] Query path
  - [ ] Inverted File (IVF)
  - [ ] Improve locality for HNSW
  - [ ] RabitQ quantization
- [ ] Indexing path
  - [ ] Support realtime indexing
## Why MuopDB?
This is an educational project for me to learn Rust & vector database.

## Building

Install prerequisites:
* Rust: https://www.rust-lang.org/tools/install
* Others
```
# macos
brew install hdf5 protobuf

export HDF5_DIR="$(brew --prefix hdf5)"
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
