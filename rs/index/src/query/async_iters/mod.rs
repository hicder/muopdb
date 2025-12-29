use anyhow::Result;

mod and_iter;
mod ids_iter;
mod or_iter;
mod term_iter;

pub use and_iter::AsyncAndIter;
pub use ids_iter::AsyncIdsIter;
pub use or_iter::AsyncOrIter;
pub use term_iter::AsyncTermIter;

#[async_trait::async_trait]
pub trait AsyncInvertedIndexIter: Send {
    async fn next(&mut self) -> Result<Option<u32>>;
    async fn skip_to(&mut self, point_id: u32) -> Result<()>;
    async fn point_id(&self) -> Option<u32>;
}

pub enum AsyncIter {
    And(AsyncAndIter),
    Or(AsyncOrIter),
    Ids(AsyncIdsIter),
    Term(AsyncTermIter),
}

#[async_trait::async_trait]
impl AsyncInvertedIndexIter for AsyncIter {
    async fn next(&mut self) -> Result<Option<u32>> {
        match self {
            AsyncIter::And(iter) => iter.next().await,
            AsyncIter::Or(iter) => iter.next().await,
            AsyncIter::Ids(iter) => iter.next().await,
            AsyncIter::Term(iter) => iter.next().await,
        }
    }

    async fn skip_to(&mut self, point_id: u32) -> Result<()> {
        match self {
            AsyncIter::And(iter) => iter.skip_to(point_id).await,
            AsyncIter::Or(iter) => iter.skip_to(point_id).await,
            AsyncIter::Ids(iter) => iter.skip_to(point_id).await,
            AsyncIter::Term(iter) => iter.skip_to(point_id).await,
        }
    }

    async fn point_id(&self) -> Option<u32> {
        match self {
            AsyncIter::And(iter) => iter.point_id().await,
            AsyncIter::Or(iter) => iter.point_id().await,
            AsyncIter::Ids(iter) => iter.point_id().await,
            AsyncIter::Term(iter) => iter.point_id().await,
        }
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    fn ids(ids: &[u32]) -> AsyncIter {
        AsyncIter::Ids(AsyncIdsIter::new(ids.to_vec()))
    }

    #[tokio::test]
    async fn test_async_or_and_ids_nested() {
        let a = ids(&[1, 2, 3, 4, 5]);
        let b = ids(&[3, 4, 5, 6, 7]);
        let c = ids(&[4, 5, 6, 7, 8]);
        let and = AsyncIter::And(AsyncAndIter::new(vec![a, b]));
        let mut iter = AsyncOrIter::new(vec![and, c]);
        let mut results = Vec::new();
        while let Some(point) = iter.next().await.unwrap() {
            results.push(point);
        }
        assert_eq!(results, vec![3, 4, 5, 6, 7, 8]);
    }

    #[tokio::test]
    async fn test_async_and_or_ids_nested() {
        let a = ids(&[1, 2, 3, 4, 5]);
        let b = ids(&[3, 4, 5, 6, 7]);
        let c = ids(&[4, 5, 6, 7, 8]);
        let or = AsyncIter::Or(AsyncOrIter::new(vec![a, b]));
        let mut and = AsyncAndIter::new(vec![or, c]);
        let mut results = Vec::new();
        while let Some(point) = and.next().await.unwrap() {
            results.push(point);
        }
        assert_eq!(results, vec![4, 5, 6, 7]);
    }
}
