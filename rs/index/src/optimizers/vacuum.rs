use anyhow::Result;
use quantization::noq::noq::NoQuantizerL2;
use quantization::quantization::Quantizer;

use super::SegmentOptimizer;
use crate::multi_spann::builder::MultiSpannBuilder;
use crate::multi_spann::writer::MultiSpannWriter;
use crate::segment::pending_segment::PendingSegment;

pub struct VacuumOptimizer<Q: Quantizer + Clone> {
    _marker: std::marker::PhantomData<Q>,
}

impl<Q: Quantizer + Clone + Send + Sync> VacuumOptimizer<Q> {
    pub fn new() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}

impl<Q: Quantizer + Clone + Send + Sync> SegmentOptimizer<Q> for VacuumOptimizer<Q> {
    #[allow(unused)]
    default fn optimize(&self, segment: &PendingSegment<Q>) -> Result<()> {
        Err(anyhow::anyhow!("not supported"))
    }
}

impl SegmentOptimizer<NoQuantizerL2> for VacuumOptimizer<NoQuantizerL2> {
    fn optimize(&self, pending_segment: &PendingSegment<NoQuantizerL2>) -> Result<()> {
        let inner_segments = pending_segment.inner_segments();
        if inner_segments.len() != 1 {
            return Ok(());
        }

        let config = pending_segment.collection_config();
        let mut builder = MultiSpannBuilder::new(config.clone(), pending_segment.base_directory())?;
        let inner_segment = &inner_segments[0];
        let all_user_ids = pending_segment.all_user_ids();

        for user_id in all_user_ids {
            let iter = inner_segment.iter_for_user(user_id);
            if let Some(iter) = iter {
                let mut iter = iter;
                while let Some((doc_id, vector)) = iter.next() {
                    builder.insert(user_id, doc_id, vector)?;
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
    use config::collection::CollectionConfig;

    use super::*;
    use crate::collection::collection::Collection;
    use crate::collection::reader::CollectionReader;
    use crate::segment::Segment;

    #[tokio::test]
    async fn test_vacuum_optimizer() -> Result<()> {
        let tmp_dir = tempdir::TempDir::new("test_vacuum_optimizer")?;
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
        collection.insert_for_users(&[0], 2, &[14.0, 15.0, 16.0], 1)?;
        collection.flush()?;

        // Now we have 1 segment, let vacuum it

        let segments = collection.get_current_toc().toc.clone();
        assert_eq!(segments.len(), 1);

        let pending_segment = collection.init_optimizing(&segments)?;

        let optimizer = VacuumOptimizer::<NoQuantizerL2>::new();
        collection.run_optimizer(&optimizer, &pending_segment)?;

        let segments = collection.get_current_toc().toc.clone();
        assert_eq!(segments.len(), 1);

        let snapshot = collection.get_snapshot()?;
        let result = snapshot
            .search_for_user(0, vec![11.0, 12.0, 13.0], 3, 10, false)
            .await
            .unwrap();
        assert_eq!(result.id_with_scores.len(), 2);

        let mut result_ids = result
            .id_with_scores
            .iter()
            .map(|id| id.doc_id)
            .collect::<Vec<_>>();
        result_ids.sort();
        assert_eq!(result_ids, vec![1, 2]);

        let result = snapshot
            .search_for_user(0, vec![11.0, 12.0, 13.0], 1, 10, false)
            .await
            .unwrap();
        assert_eq!(result.id_with_scores.len(), 1);
        let mut result_ids = result
            .id_with_scores
            .iter()
            .map(|id| id.doc_id)
            .collect::<Vec<_>>();
        result_ids.sort();
        assert_eq!(result_ids, vec![2]);

        Ok(())
    }

    #[tokio::test]
    async fn test_vacuum_invalidated_optimizer() -> Result<()> {
        let tmp_dir = tempdir::TempDir::new("test_vacuum_invalidated_optimizer")?;
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
        collection.insert_for_users(&[0], 2, &[20.0, 21.0, 22.0], 1)?;
        collection.insert_for_users(&[0], 3, &[27.0, 28.0, 29.0], 2)?;
        collection.flush()?;

        // Now we have 1 segment, let vacuum it

        let segments = collection.get_current_toc().toc.clone();
        assert_eq!(segments.len(), 1);

        // Remove a doc from the first segment
        assert!(collection
            .all_segments()
            .get(&segments[0])
            .unwrap()
            .value()
            .remove(0, 2)
            .is_ok());

        let pending_segment = collection.init_optimizing(&segments)?;

        let optimizer = VacuumOptimizer::<NoQuantizerL2>::new();
        collection.run_optimizer(&optimizer, &pending_segment)?;

        let segments = collection.get_current_toc().toc.clone();
        assert_eq!(segments.len(), 1);

        let snapshot = collection.get_snapshot()?;
        let result = snapshot
            .search_for_user(0, vec![20.0, 21.0, 22.0], 1, 10, false)
            .await
            .unwrap();
        assert_eq!(result.id_with_scores.len(), 1);

        let mut result_ids = result
            .id_with_scores
            .iter()
            .map(|id| id.doc_id)
            .collect::<Vec<_>>();
        result_ids.sort();
        assert_eq!(result_ids, vec![3]);

        Ok(())
    }

    #[tokio::test]
    async fn test_vacuum_optimizer_with_multiple_users() -> Result<()> {
        let tmp_dir = tempdir::TempDir::new("test_vacuum_optimizer_with_multiple_users")?;
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

        collection.insert_for_users(&[0], 101, &[1.0, 2.0, 3.0], 0)?;
        collection.insert_for_users(&[1], 202, &[14.0, 15.0, 16.0], 1)?;
        collection.flush()?;

        // Now we have 1 segment, let vacuum it

        let segments = collection.get_current_toc().toc.clone();
        assert_eq!(segments.len(), 1);

        let pending_segment = collection.init_optimizing(&segments)?;

        let optimizer = VacuumOptimizer::<NoQuantizerL2>::new();
        collection.run_optimizer(&optimizer, &pending_segment)?;

        let segments = collection.get_current_toc().toc.clone();
        assert_eq!(segments.len(), 1);

        let snapshot = collection.get_snapshot()?;
        let result = snapshot
            .search_for_user(0, vec![11.0, 12.0, 13.0], 3, 10, false)
            .await
            .unwrap();
        assert_eq!(result.id_with_scores.len(), 1);

        let mut result_ids = result
            .id_with_scores
            .iter()
            .map(|id| id.doc_id)
            .collect::<Vec<_>>();
        result_ids.sort();
        assert_eq!(result_ids, vec![101]);

        let result = snapshot
            .search_for_user(1, vec![11.0, 12.0, 13.0], 3, 10, false)
            .await
            .unwrap();
        assert_eq!(result.id_with_scores.len(), 1);

        let mut result_ids = result
            .id_with_scores
            .iter()
            .map(|id| id.doc_id)
            .collect::<Vec<_>>();
        result_ids.sort();
        assert_eq!(result_ids, vec![202]);

        Ok(())
    }

    #[tokio::test]
    async fn test_vacuum_invalidated_optimizer_with_multiple_users() -> Result<()> {
        let tmp_dir =
            tempdir::TempDir::new("test_vacuum_invalidated_optimizer_with_multiple_users")?;
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

        collection.insert_for_users(&[0], 11, &[1.0, 2.0, 3.0], 0)?;
        collection.insert_for_users(&[0], 12, &[20.0, 21.0, 22.0], 1)?;
        collection.insert_for_users(&[0], 13, &[27.0, 28.0, 29.0], 2)?;

        collection.insert_for_users(&[1], 201, &[1.0, 2.0, 3.0], 0)?;
        collection.insert_for_users(&[1], 202, &[20.0, 21.0, 22.0], 1)?;
        collection.insert_for_users(&[1], 203, &[27.0, 28.0, 29.0], 2)?;
        collection.flush()?;

        // Now we have 1 segment, let vacuum it

        let segments = collection.get_current_toc().toc.clone();
        assert_eq!(segments.len(), 1);

        // Remove a doc from the first segment
        assert!(collection
            .all_segments()
            .get(&segments[0])
            .unwrap()
            .value()
            .remove(1, 203)
            .is_ok());

        let pending_segment = collection.init_optimizing(&segments)?;

        let optimizer = VacuumOptimizer::<NoQuantizerL2>::new();
        collection.run_optimizer(&optimizer, &pending_segment)?;

        let segments = collection.get_current_toc().toc.clone();
        assert_eq!(segments.len(), 1);

        let snapshot = collection.get_snapshot()?;
        let result = snapshot
            .search_for_user(0, vec![20.0, 21.0, 22.0], 3, 10, false)
            .await
            .unwrap();
        assert_eq!(result.id_with_scores.len(), 3);

        let mut result_ids = result
            .id_with_scores
            .iter()
            .map(|id| id.doc_id)
            .collect::<Vec<_>>();
        result_ids.sort();
        assert_eq!(result_ids, vec![11, 12, 13]);

        let result = snapshot
            .search_for_user(1, vec![20.0, 21.0, 22.0], 3, 10, false)
            .await
            .unwrap();
        assert_eq!(result.id_with_scores.len(), 2);

        let mut result_ids = result
            .id_with_scores
            .iter()
            .map(|id| id.doc_id)
            .collect::<Vec<_>>();
        result_ids.sort();
        assert_eq!(result_ids, vec![201, 202]);

        Ok(())
    }
}
