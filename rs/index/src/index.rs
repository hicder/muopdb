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

    #[allow(unused_variables)]
    fn search_with_id(
        &self,
        id: u128,
        query: &[f32],
        k: usize,
        ef_construction: u32,
        context: &mut SearchContext,
    ) -> Option<Vec<IdWithScore>> {
        // This is a default implementation. In MultiSpann, we will override this function.
        self.search(query, k, ef_construction, context)
    }
}

pub type BoxedSearchable = Box<dyn Searchable + Send + Sync>;
