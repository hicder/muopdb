use anyhow::Result;
use quantization::noq::noq::NoQuantizerL2;
use quantization::quantization::Quantizer;

use super::SegmentOptimizer;
use crate::multi_spann::builder::MultiSpannBuilder;
use crate::multi_spann::writer::MultiSpannWriter;
use crate::segment::pending_segment::PendingSegment;

pub struct MergeOptimizer<Q: Quantizer + Clone> {
    _marker: std::marker::PhantomData<Q>,
}

impl<Q: Quantizer + Clone + Send + Sync> MergeOptimizer<Q> {
    pub fn new() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}

impl<Q: Quantizer + Clone + Send + Sync> SegmentOptimizer<Q> for MergeOptimizer<Q> {
    #[allow(unused)]
    default fn optimize(&self, segment: &PendingSegment<Q>) -> Result<()> {
        Err(anyhow::anyhow!("not supported"))
    }
}

impl SegmentOptimizer<NoQuantizerL2> for MergeOptimizer<NoQuantizerL2> {
    fn optimize(&self, pending_segment: &PendingSegment<NoQuantizerL2>) -> Result<()> {
        let inner_segments = pending_segment.inner_segments();
        let all_user_ids = pending_segment.all_user_ids();

        let config = pending_segment.collection_config();
        let mut builder = MultiSpannBuilder::new(config.clone(), pending_segment.base_directory())?;

        for user_id in all_user_ids {
            for inner_segment in inner_segments {
                let iter = inner_segment.iter_for_user(user_id);
                if let Some(iter) = iter {
                    let mut iter = iter;
                    while let Some((doc_id, vector)) = iter.next() {
                        builder.insert(user_id, doc_id, vector)?;
                    }
                }
            }
        }

        builder.build()?;
        let writer = MultiSpannWriter::new(pending_segment.base_directory().clone());
        writer.write(&mut builder)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use config::collection::CollectionConfig;

    use super::*;
    use crate::collection::collection::Collection;
    use crate::collection::reader::CollectionReader;

    #[tokio::test]
    async fn test_merge_optimizer() -> Result<()> {
        let tmp_dir = tempdir::TempDir::new("test_merge_optimizer")?;
        // Create directory if it doesn't exist
        std::fs::create_dir_all(&tmp_dir)?;

        let base_directory = tmp_dir.path().to_str().unwrap().to_string();
        let mut config = CollectionConfig::default_test_config();
        config.num_features = 3;
        // Don't use WAL for this test
        // Also, don't flush automatically
        config.wal_file_size = 0;
        config.max_time_to_flush_ms = 0;
        config.max_pending_ops = 0;
        config.initial_num_centroids = 1;

        Collection::<NoQuantizerL2>::init_new_collection(base_directory.clone(), &config)?;

        let reader = CollectionReader::new(base_directory.clone());
        let collection = reader.read::<NoQuantizerL2>()?;

        collection.insert_for_users(&[0], 1, &[1.0, 2.0, 3.0], 0)?;
        collection.insert_for_users(&[0], 2, &[4.0, 5.0, 6.0], 1)?;
        collection.insert_for_users(&[0], 3, &[7.0, 8.0, 9.0], 2)?;

        collection.flush()?;

        collection.insert_for_users(&[0], 4, &[100.0, 101.0, 102.0], 3)?;
        collection.insert_for_users(&[0], 5, &[103.0, 104.0, 105.0], 4)?;
        collection.insert_for_users(&[0], 6, &[106.0, 107.0, 108.0], 5)?;

        collection.flush()?;

        // Now we have 2 segments, let merge them

        let segments = collection.get_current_toc().toc.clone();
        assert_eq!(segments.len(), 2);
        let pending_segment = collection.init_optimizing(&segments)?;

        let optimizer = MergeOptimizer::<NoQuantizerL2>::new();
        collection.run_optimizer(&optimizer, &pending_segment)?;

        let segments = collection.get_current_toc().toc.clone();
        assert_eq!(segments.len(), 1);

        let snapshot = collection.get_snapshot()?;
        let snapshot = Arc::new(snapshot);
        let result = snapshot
            .search_with_id(0, vec![100.0, 101.0, 102.0], 3, 10, false)
            .await
            .unwrap();
        assert_eq!(result.id_with_scores.len(), 3);

        let mut result_ids = result
            .id_with_scores
            .iter()
            .map(|id| id.id)
            .collect::<Vec<_>>();
        result_ids.sort();
        assert_eq!(result_ids, vec![4, 5, 6]);

        let result = snapshot
            .search_with_id(0, vec![1.0, 2.0, 3.0], 3, 10, false)
            .await
            .unwrap();
        assert_eq!(result.id_with_scores.len(), 3);
        let mut result_ids = result
            .id_with_scores
            .iter()
            .map(|id| id.id)
            .collect::<Vec<_>>();
        result_ids.sort();
        assert_eq!(result_ids, vec![1, 2, 3]);

        Ok(())
    }

    #[tokio::test]
    async fn test_merge_optimizer_with_multiple_users() -> Result<()> {
        let tmp_dir = tempdir::TempDir::new("test_merge_optimizer")?;
        // Create directory if it doesn't exist
        std::fs::create_dir_all(&tmp_dir)?;

        let base_directory = tmp_dir.path().to_str().unwrap().to_string();
        let mut config = CollectionConfig::default_test_config();
        config.num_features = 3;
        config.initial_num_centroids = 1;
        // Don't use WAL for this test
        // Also, don't flush automatically
        config.wal_file_size = 0;
        config.max_time_to_flush_ms = 0;
        config.max_pending_ops = 0;

        Collection::<NoQuantizerL2>::init_new_collection(base_directory.clone(), &config)?;

        let reader = CollectionReader::new(base_directory.clone());
        let collection = reader.read::<NoQuantizerL2>()?;

        collection.insert_for_users(&[0], 1, &[1.0, 2.0, 3.0], 0)?;
        collection.insert_for_users(&[0], 2, &[4.0, 5.0, 6.0], 1)?;
        collection.insert_for_users(&[0], 3, &[7.0, 8.0, 9.0], 2)?;

        collection.flush()?;

        collection.insert_for_users(&[1], 4, &[10.0, 11.0, 12.0], 3)?;
        collection.insert_for_users(&[1], 5, &[13.0, 14.0, 15.0], 4)?;
        collection.insert_for_users(&[1], 6, &[16.0, 17.0, 18.0], 5)?;

        collection.flush()?;

        // Now we have 2 segments, let merge them

        let segments = collection.get_current_toc().toc.clone();
        assert_eq!(segments.len(), 2);
        let pending_segment = collection.init_optimizing(&segments)?;

        let optimizer = MergeOptimizer::<NoQuantizerL2>::new();
        collection.run_optimizer(&optimizer, &pending_segment)?;

        let segments = collection.get_current_toc().toc.clone();
        assert_eq!(segments.len(), 1);

        let snapshot = collection.get_snapshot()?;
        let result = snapshot
            .search_with_id(0, vec![1.0, 2.0, 3.0], 3, 10, false)
            .await
            .unwrap();
        assert_eq!(result.id_with_scores.len(), 3);

        let mut result_ids = result
            .id_with_scores
            .iter()
            .map(|id| id.id)
            .collect::<Vec<_>>();
        result_ids.sort();
        assert_eq!(result_ids, vec![1, 2, 3]);

        let result = snapshot
            .search_with_id(1, vec![10.0, 11.0, 12.0], 3, 10, false)
            .await
            .unwrap();
        assert_eq!(result.id_with_scores.len(), 3);

        let mut result_ids = result
            .id_with_scores
            .iter()
            .map(|id| id.id)
            .collect::<Vec<_>>();
        result_ids.sort();
        assert_eq!(result_ids, vec![4, 5, 6]);

        Ok(())
    }
}
