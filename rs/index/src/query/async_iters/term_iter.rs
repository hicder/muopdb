use anyhow::Result;
use compression::compression::AsyncIntSeqIterator;
use compression::elias_fano::block_based_decoder::BlockBasedEliasFanoIterator;

use crate::query::async_iters::AsyncInvertedIndexIter;
use crate::query::iters::IterState;

pub struct AsyncTermIter {
    iter: BlockBasedEliasFanoIterator<u32>,
    state: IterState<u32>,
}

impl AsyncTermIter {
    pub fn new(iter: BlockBasedEliasFanoIterator<u32>) -> Self {
        Self {
            iter,
            state: IterState::NotStarted,
        }
    }
}

#[async_trait::async_trait]
impl AsyncInvertedIndexIter for AsyncTermIter {
    async fn next(&mut self) -> Result<Option<u32>> {
        match self.state {
            IterState::NotStarted => {
                let point = self.iter.current().await?;
                self.iter.next().await?;
                self.state = if let Some(p) = point {
                    IterState::At(p)
                } else {
                    IterState::Exhausted
                };
                Ok(point)
            }
            IterState::At(_) => {
                let point = self.iter.next().await?;
                self.state = if let Some(p) = point {
                    IterState::At(p)
                } else {
                    IterState::Exhausted
                };
                Ok(point)
            }
            IterState::Exhausted => Ok(None),
        }
    }

    async fn skip_to(&mut self, point_id: u32) -> Result<()> {
        self.iter.skip_to(point_id).await?;
        let point = self.iter.current().await?;
        self.state = if let Some(p) = point {
            IterState::At(p)
        } else {
            IterState::Exhausted
        };
        Ok(())
    }

    async fn point_id(&self) -> Option<u32> {
        match self.state {
            IterState::At(id) => Some(id),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use tempdir::TempDir;
    use utils::file_io::env::{DefaultEnv, Env, EnvConfig, FileType};

    use super::*;
    use crate::terms::builder::TermBuilder;
    use crate::terms::index::TermIndex;
    use crate::terms::writer::TermWriter;

    async fn setup_term_index(base_dir_str: String) -> TermIndex {
        let mut builder = TermBuilder::new().unwrap();
        builder.add(0, "apple".to_string()).unwrap();
        builder.add(1, "apple".to_string()).unwrap();
        builder.add(2, "apple".to_string()).unwrap();
        builder.add(3, "apple".to_string()).unwrap();
        builder.add(5, "apple".to_string()).unwrap();
        builder.add(0, "banana".to_string()).unwrap();
        builder.add(2, "banana".to_string()).unwrap();

        builder.build().unwrap();
        let writer = TermWriter::new(base_dir_str.clone());
        writer.write(&mut builder).unwrap();

        let path = format!("{base_dir_str}/combined");
        let file_len = std::fs::metadata(&path).unwrap().len();

        let config = EnvConfig {
            file_type: FileType::CachedStandard,
            ..EnvConfig::default()
        };
        let env = Arc::new(DefaultEnv::new(config));
        let file_io = env.open(&path).await.unwrap().file_io;

        TermIndex::new_with_file_io(file_io, path, 0, file_len as usize)
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn test_async_term_iter_next() {
        let temp_dir = TempDir::new("test_async_term_iter").unwrap();
        let base_dir_str = temp_dir.path().to_str().unwrap().to_string();

        let index = setup_term_index(base_dir_str).await;
        let apple_id = index.get_term_id("apple").unwrap();
        let it = index
            .get_posting_list_iterator_block_based(apple_id)
            .await
            .unwrap();
        let mut iter = AsyncTermIter::new(it);

        assert_eq!(iter.next().await.unwrap(), Some(0));
        assert_eq!(iter.next().await.unwrap(), Some(1));
        assert_eq!(iter.next().await.unwrap(), Some(2));
        assert_eq!(iter.next().await.unwrap(), Some(3));
        assert_eq!(iter.next().await.unwrap(), Some(5));
        assert_eq!(iter.next().await.unwrap(), None);
    }

    #[tokio::test]
    async fn test_async_term_iter_skip_to() {
        let temp_dir = TempDir::new("test_async_term_iter_skip").unwrap();
        let base_dir_str = temp_dir.path().to_str().unwrap().to_string();

        let index = setup_term_index(base_dir_str).await;
        let apple_id = index.get_term_id("apple").unwrap();
        let it = index
            .get_posting_list_iterator_block_based(apple_id)
            .await
            .unwrap();
        let mut iter = AsyncTermIter::new(it);

        iter.next().await.unwrap(); // 0
        iter.skip_to(2).await.unwrap();
        assert_eq!(iter.point_id().await, Some(2));
        assert_eq!(iter.next().await.unwrap(), Some(3));

        iter.skip_to(5).await.unwrap();
        assert_eq!(iter.point_id().await, Some(5));
        assert_eq!(iter.next().await.unwrap(), None);
    }

    #[tokio::test]
    async fn test_async_term_iter_point_id() {
        let temp_dir = TempDir::new("test_async_term_iter_point_id").unwrap();
        let base_dir_str = temp_dir.path().to_str().unwrap().to_string();

        let index = setup_term_index(base_dir_str).await;
        let apple_id = index.get_term_id("apple").unwrap();
        let it = index
            .get_posting_list_iterator_block_based(apple_id)
            .await
            .unwrap();
        let mut iter = AsyncTermIter::new(it);

        assert_eq!(iter.point_id().await, None);
        iter.next().await.unwrap();
        assert_eq!(iter.point_id().await, Some(0));
        iter.next().await.unwrap();
        assert_eq!(iter.point_id().await, Some(1));
    }
}
