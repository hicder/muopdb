use anyhow::Result;

use crate::query::async_iters::{AsyncInvertedIndexIter, AsyncIter};
use crate::query::iters::IterState;

pub struct AsyncAndIter {
    iters: Vec<AsyncIter>,
    state: IterState<u32>,
}

impl AsyncAndIter {
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

    async fn align(&mut self) -> Result<Option<u32>> {
        loop {
            let mut max_point = None;

            for child in self.iters.iter() {
                match child.point_id().await {
                    Some(point) => {
                        max_point = Some(max_point.map_or(point, |m: u32| m.max(point)));
                    }
                    None => return Ok(None),
                }
            }

            let target = match max_point {
                Some(t) => t,
                None => return Ok(None),
            };

            let mut all_equal = true;
            for child in self.iters.iter_mut() {
                if child.point_id().await == Some(target) {
                    continue;
                }
                child.skip_to(target).await?;
                match child.point_id().await {
                    Some(point) if point == target => {}
                    Some(_) => {
                        all_equal = false;
                    }
                    None => {
                        return Ok(None);
                    }
                }
            }

            if all_equal {
                return Ok(Some(target));
            }
        }
    }
}

#[async_trait::async_trait]
impl AsyncInvertedIndexIter for AsyncAndIter {
    async fn next(&mut self) -> Result<Option<u32>> {
        match self.state {
            IterState::NotStarted => {
                for child in self.iters.iter_mut() {
                    if child.next().await?.is_none() {
                        self.state = IterState::Exhausted;
                        return Ok(None);
                    }
                }
                match self.align().await? {
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
            IterState::At(_) => {
                if self.iters[0].next().await?.is_none() {
                    self.state = IterState::Exhausted;
                    return Ok(None);
                }
                match self.align().await? {
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
            IterState::Exhausted => Ok(None),
        }
    }

    async fn skip_to(&mut self, target: u32) -> Result<()> {
        match self.state {
            IterState::Exhausted => Ok(()),
            _ => {
                for child in self.iters.iter_mut() {
                    child.skip_to(target).await?;
                    if child.point_id().await.is_none() {
                        self.state = IterState::Exhausted;
                        return Ok(());
                    }
                }
                match self.align().await? {
                    Some(point) => self.state = IterState::At(point),
                    None => self.state = IterState::Exhausted,
                }
                Ok(())
            }
        }
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
    async fn test_async_and_iter_basic_intersection() {
        let a = ids(&[1, 3, 5, 7]);
        let b = ids(&[3, 4, 5, 7, 8]);
        let c = ids(&[2, 3, 5, 7, 9]);
        let mut iter = AsyncAndIter::new(vec![a, b, c]);
        let mut results = Vec::new();
        while let Some(point) = iter.next().await.unwrap() {
            results.push(point);
        }
        assert_eq!(results, vec![3, 5, 7]);
    }

    #[tokio::test]
    async fn test_async_and_iter_empty() {
        let a = ids(&[1, 2, 3]);
        let b = ids(&[]);
        let mut iter = AsyncAndIter::new(vec![a, b]);
        assert_eq!(iter.next().await.unwrap(), None);
    }

    #[tokio::test]
    async fn test_async_and_iter_single_child() {
        let a = ids(&[1, 2, 3]);
        let mut iter = AsyncAndIter::new(vec![a]);
        assert_eq!(iter.next().await.unwrap(), Some(1));
        assert_eq!(iter.next().await.unwrap(), Some(2));
        assert_eq!(iter.next().await.unwrap(), Some(3));
        assert_eq!(iter.next().await.unwrap(), None);
    }

    #[tokio::test]
    async fn test_async_and_iter_all_empty() {
        let mut iter = AsyncAndIter::new(vec![]);
        assert_eq!(iter.next().await.unwrap(), None);
    }

    #[tokio::test]
    async fn test_async_and_iter_no_intersection() {
        let a = ids(&[1, 2, 3]);
        let b = ids(&[4, 5, 6]);
        let mut iter = AsyncAndIter::new(vec![a, b]);
        assert_eq!(iter.next().await.unwrap(), None);
    }

    #[tokio::test]
    async fn test_async_and_iter_superset() {
        let a = ids(&[1, 2, 3, 4, 5]);
        let b = ids(&[2, 4]);
        let mut iter = AsyncAndIter::new(vec![a, b]);
        let mut results = Vec::new();
        while let Some(point) = iter.next().await.unwrap() {
            results.push(point);
        }
        assert_eq!(results, vec![2, 4]);
    }

    #[tokio::test]
    async fn test_async_and_iter_all_empty_children() {
        let mut iter = AsyncAndIter::new(vec![ids(&[]), ids(&[])]);
        assert_eq!(iter.next().await.unwrap(), None);
    }
}
