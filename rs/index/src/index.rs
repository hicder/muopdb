use crate::utils::SearchContext;

#[derive(Debug)]
pub struct IdWithScore {
    pub id: u64,
    pub score: f32,
}

/// Main trait for index
pub trait Index {
    /// Search for the nearest neighbors of a query vector
    fn search(
        &self,
        query: &[f32],
        k: usize,
        context: &mut SearchContext,
    ) -> Option<Vec<IdWithScore>>;
}

pub type BoxedIndex = Box<dyn Index + Send + Sync>;
