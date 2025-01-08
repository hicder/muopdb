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

    fn search_with_id(
        &self,
        _id: u64,
        query: &[f32],
        k: usize,
        ef_construction: u32,
        context: &mut SearchContext,
    ) -> Option<Vec<IdWithScore>> {
        self.search(query, k, ef_construction, context)
    }
}

pub type BoxedSearchable = Box<dyn Searchable + Send + Sync>;
