//! Main trait for index
//! TODO(hicder): Add more methods
pub trait Index {
    /// Search for the nearest neighbors of a query vector
    fn search(&self, query: &[f32], k: usize) -> Option<Vec<u64>>;
}

pub type BoxedIndex = Box<dyn Index + Send + Sync>;
