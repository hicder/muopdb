//! Main trait for index
//! TODO(hicder): Add more methods
pub trait Index {
    /// Search for the nearest neighbors of a query vector
    fn search(&self, query: &[f32]) -> Option<Vec<u64>>;
}
