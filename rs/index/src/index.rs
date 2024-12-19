use crate::utils::{IdWithScore, SearchContext};

/// Main trait for index
pub trait Searchable {
    /// Search for the nearest neighbors of a query vector
    fn search(
        &self,
        query: &[f32],
        k: usize,
        ef_construction: u32,
        context: &mut SearchContext,
    ) -> Option<Vec<IdWithScore>>;
}

pub type BoxedSearchable = Box<dyn Searchable + Send + Sync>;
