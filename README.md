# MuopDB - A vector database for machine learning

## Introduction

MuopDB is a vector database for machine learning. This project is done under [TechCare Coaching](https://techcarecoaching.com/). It plans to support the following features:

- [ ] Query path
  - [x] Vector similarity search
  - [ ] Inverted File (IVF)
  - [x] Hierarchical Navigable Small Worlds (HNSW)
  - [x] Product Quantization (PQ)
- [ ] Indexing path
  - [x] Support periodic offline indexing
  - [ ] Support realtime indexing
- [ ] Database Management
  - [ ] Doc-sharding & query fan-out with aggregator-leaf architecture
  - [x] In-memory & disk-based storage with mmap

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
cargo build
```
