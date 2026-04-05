#![feature(auto_traits)]
#![feature(min_specialization)]

pub mod bm25;
pub mod collection;
pub mod hnsw;
pub mod ivf;
pub mod multi_spann;
pub mod multi_terms;
pub mod optimizers;
pub mod posting_list;
pub mod query;
pub mod segment;
pub mod spann;
pub mod terms;
pub mod tokenizer;
pub mod traverse_state;
pub mod utils;
pub mod vector;
pub mod wal;

// Re-export key BM25 types at crate level for convenience
pub use bm25::{BM25Config, BM25Scorer};
pub use bm25::{Bm25Query, Bm25SearchResult, Bm25SearchResults};
pub use bm25::Bm25Searcher;
pub use bm25::TopKSelector;
