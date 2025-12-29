use anyhow::Result;

use crate::query::async_iters::{AsyncInvertedIndexIter, AsyncIter};
use crate::query::iters::IterState;

pub struct AsyncOrIter {
    iters: Vec<AsyncIter>,
    state: IterState<u32>,
}

impl AsyncOrIter {
    pub fn new(iters: Vec<AsyncIter>) -> Self {
        if iters.is_empty() {
            return Self {
                iters,
                state: IterState::Exhausted,
            };
        }
        Self {
            iters,
            state: IterState::NotStarted,
        }
    }

    async fn min_point(&self) -> Option<u32> {
        let mut min = None;
        for child in self.iters.iter() {
            if let Some(p) = child.point_id().await {
                min = Some(min.map_or(p, |m: u32| m.min(p)));
            }
        }
        min
    }

    async fn advance_consumed(&mut self, point: u32) -> Result<()> {
        for child in self.iters.iter_mut() {
            if child.point_id().await == Some(point) {
                child.next().await?;
            }
        }
        Ok(())
    }
}

#[async_trait::async_trait]
impl AsyncInvertedIndexIter for AsyncOrIter {
    async fn next(&mut self) -> Result<Option<u32>> {
        match self.state {
            IterState::NotStarted => {
                for child in self.iters.iter_mut() {
                    child.next().await?;
                }
            }
            IterState::At(prev_point) => {
                self.advance_consumed(prev_point).await?;
            }
            IterState::Exhausted => return Ok(None),
        }

        match self.min_point().await {
            Some(point) => {
                self.state = IterState::At(point);
                Ok(Some(point))
            }
            None => {
                self.state = IterState::Exhausted;
                Ok(None)
            }
        }
    }

    async fn skip_to(&mut self, target: u32) -> Result<()> {
        if matches!(self.state, IterState::Exhausted) {
            return Ok(());
        }

        for child in self.iters.iter_mut() {
            child.skip_to(target).await?;
        }

        match self.min_point().await {
            Some(point) => self.state = IterState::At(point),
            None => self.state = IterState::Exhausted,
        }
        Ok(())
    }

    async fn point_id(&self) -> Option<u32> {
        match self.state {
            IterState::At(point) => Some(point),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(ids: &[u32]) -> AsyncIter {
        AsyncIter::Ids(crate::query::async_iters::ids_iter::AsyncIdsIter::new(
            ids.to_vec(),
        ))
    }

    #[tokio::test]
    async fn test_async_or_iter_basic_union() {
        let a = ids(&[1, 3, 5, 7]);
        let b = ids(&[3, 4, 5, 7, 8]);
        let c = ids(&[2, 3, 5, 7, 9]);
        let mut iter = AsyncOrIter::new(vec![a, b, c]);
        let mut results = Vec::new();
        while let Some(point) = iter.next().await.unwrap() {
            results.push(point);
        }
        assert_eq!(results, vec![1, 2, 3, 4, 5, 7, 8, 9]);
    }

    #[tokio::test]
    async fn test_async_or_iter_empty() {
        let a = ids(&[1, 2, 3]);
        let b = ids(&[]);
        let mut iter = AsyncOrIter::new(vec![a, b]);
        let mut results = Vec::new();
        while let Some(point) = iter.next().await.unwrap() {
            results.push(point);
        }
        assert_eq!(results, vec![1, 2, 3]);
    }

    #[tokio::test]
    async fn test_async_or_iter_single_child() {
        let a = ids(&[1, 2, 3]);
        let mut iter = AsyncOrIter::new(vec![a]);
        assert_eq!(iter.next().await.unwrap(), Some(1));
        assert_eq!(iter.next().await.unwrap(), Some(2));
        assert_eq!(iter.next().await.unwrap(), Some(3));
        assert_eq!(iter.next().await.unwrap(), None);
    }

    #[tokio::test]
    async fn test_async_or_iter_all_empty() {
        let mut iter = AsyncOrIter::new(vec![]);
        assert_eq!(iter.next().await.unwrap(), None);
    }

    #[tokio::test]
    async fn test_async_or_iter_no_overlap() {
        let a = ids(&[1, 2, 3]);
        let b = ids(&[4, 5, 6]);
        let mut iter = AsyncOrIter::new(vec![a, b]);
        let mut results = Vec::new();
        while let Some(point) = iter.next().await.unwrap() {
            results.push(point);
        }
        assert_eq!(results, vec![1, 2, 3, 4, 5, 6]);
    }
}
