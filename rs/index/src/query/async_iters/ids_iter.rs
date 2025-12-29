use anyhow::Result;

use crate::query::async_iters::AsyncInvertedIndexIter;
use crate::query::iters::IterState;

pub struct AsyncIdsIter {
    ids: Vec<u32>,
    state: IterState<usize>,
}

impl AsyncIdsIter {
    pub fn new(ids: Vec<u32>) -> Self {
        Self {
            ids,
            state: IterState::NotStarted,
        }
    }
}

#[async_trait::async_trait]
impl AsyncInvertedIndexIter for AsyncIdsIter {
    async fn next(&mut self) -> Result<Option<u32>> {
        match self.state {
            IterState::NotStarted => {
                if self.ids.is_empty() {
                    self.state = IterState::Exhausted;
                    Ok(None)
                } else {
                    self.state = IterState::At(0);
                    Ok(Some(self.ids[0]))
                }
            }
            IterState::At(idx) => {
                let next_idx = idx + 1;
                if next_idx < self.ids.len() {
                    self.state = IterState::At(next_idx);
                    Ok(Some(self.ids[next_idx]))
                } else {
                    self.state = IterState::Exhausted;
                    Ok(None)
                }
            }
            IterState::Exhausted => Ok(None),
        }
    }

    async fn skip_to(&mut self, point_id: u32) -> Result<()> {
        let start_idx = match self.state {
            IterState::NotStarted => 0,
            IterState::At(idx) => idx,
            IterState::Exhausted => return Ok(()),
        };

        for i in start_idx..self.ids.len() {
            if self.ids[i] >= point_id {
                self.state = IterState::At(i);
                return Ok(());
            }
        }
        self.state = IterState::Exhausted;
        Ok(())
    }

    async fn point_id(&self) -> Option<u32> {
        match self.state {
            IterState::At(idx) => Some(self.ids[idx]),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_async_ids_iter_basic() {
        let mut iter = AsyncIdsIter::new(vec![1, 2, 3]);
        assert_eq!(iter.next().await.unwrap(), Some(1));
        assert_eq!(iter.next().await.unwrap(), Some(2));
        assert_eq!(iter.next().await.unwrap(), Some(3));
        assert_eq!(iter.next().await.unwrap(), None);
    }

    #[tokio::test]
    async fn test_async_ids_iter_skip_to() {
        let mut iter = AsyncIdsIter::new(vec![1, 3, 5, 7, 9]);
        iter.skip_to(4).await.unwrap();
        assert_eq!(iter.point_id().await, Some(5));
        iter.skip_to(8).await.unwrap();
        assert_eq!(iter.point_id().await, Some(9));
        iter.skip_to(10).await.unwrap();
        assert_eq!(iter.next().await.unwrap(), None);
    }
}
