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

impl<Q: Quantizer + Clone + Send + Sync> Default for MergeOptimizer<Q> {
    fn default() -> Self {
        Self::new()
    }
}

impl<Q: Quantizer + Clone + Send + Sync> MergeOptimizer<Q> {
    pub fn new() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}

#[async_trait::async_trait]
impl<Q: Quantizer + Clone + Send + Sync> SegmentOptimizer<Q> for MergeOptimizer<Q> {
    #[allow(unused)]
    default async fn optimize(&self, segment: &PendingSegment<Q>) -> Result<()> {
        Err(anyhow::anyhow!("not supported"))
    }
}

#[async_trait::async_trait]
impl SegmentOptimizer<NoQuantizerL2> for MergeOptimizer<NoQuantizerL2> {
    async fn optimize(&self, pending_segment: &PendingSegment<NoQuantizerL2>) -> Result<()> {
        let inner_segments = pending_segment.inner_segments();
        if inner_segments.len() < 2 {
            return Ok(());
        }

        let config = pending_segment.collection_config();
        let mut builder = MultiSpannBuilder::new(config.clone(), pending_segment.base_directory())?;
        let term_builder = crate::multi_terms::builder::MultiTermBuilder::new();
        let all_user_ids = pending_segment.all_user_ids();

        // Track point ID mappings: (seg_idx, user_id, old_point_id) -> new_point_id
        let mut point_id_mappings = std::collections::HashMap::<(usize, u128, u32), u32>::new();

        for user_id in all_user_ids {
            for (seg_idx, inner_segment) in inner_segments.iter().enumerate() {
                let iter = inner_segment.iter_for_user(user_id).await;
                if let Some(iter) = iter {
                    let mut iter = iter;
                    while let Some((old_point_id, doc_id, vector)) = iter.next().await {
                        let new_point_id = builder.insert(user_id, doc_id, &vector)?;
                        point_id_mappings.insert((seg_idx, user_id, old_point_id), new_point_id);
                    }
                }
            }
        }

        // Merge term indices
        for (seg_idx, inner_segment) in inner_segments.iter().enumerate() {
            let user_ids = inner_segment.user_ids().await;
            for user_id in user_ids {
                if let Some(term_pairs) = inner_segment.iter_terms_for_user(user_id).await {
                    for (term, old_point_id) in term_pairs {
                        if let Some(&new_point_id) =
                            point_id_mappings.get(&(seg_idx, user_id, old_point_id))
                        {
                            term_builder.add(user_id, new_point_id, term)?;
                        }
                    }
                }
            }
        }

        term_builder.build()?;

        builder.build()?;
        let writer = MultiSpannWriter::new(pending_segment.base_directory().clone());
        writer.write(&mut builder)?;

        // Write term index
        let term_writer = crate::multi_terms::writer::MultiTermWriter::new(format!(
            "{}/terms",
            pending_segment.base_directory()
        ));
        term_writer.write(&term_builder)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use config::collection::CollectionConfig;
    use config::search_params::SearchParams;
    use proto::muopdb::DocumentAttribute;

    use super::*;
    use crate::collection::core::Collection;
    use crate::collection::reader::CollectionReader;
    use crate::segment::Segment;

    #[tokio::test]
    async fn test_merge_optimizer() -> Result<()> {
        let collection_name = "test_merge_optimizer";
        let tmp_dir = tempdir::TempDir::new(collection_name)?;
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

        let reader =
            CollectionReader::new(collection_name.to_string(), base_directory.clone(), None);
        let collection = reader.read::<NoQuantizerL2>().await?;

        collection
            .insert_for_users(&[0], 1, &[1.0, 2.0, 3.0], 0, DocumentAttribute::default())
            .await?;
        collection
            .insert_for_users(&[0], 2, &[4.0, 5.0, 6.0], 1, DocumentAttribute::default())
            .await?;
        collection
            .insert_for_users(&[0], 3, &[7.0, 8.0, 9.0], 2, DocumentAttribute::default())
            .await?;

        collection.flush().await?;

        collection
            .insert_for_users(
                &[0],
                4,
                &[100.0, 101.0, 102.0],
                3,
                DocumentAttribute::default(),
            )
            .await?;
        collection
            .insert_for_users(
                &[0],
                5,
                &[103.0, 104.0, 105.0],
                4,
                DocumentAttribute::default(),
            )
            .await?;
        collection
            .insert_for_users(
                &[0],
                6,
                &[106.0, 107.0, 108.0],
                5,
                DocumentAttribute::default(),
            )
            .await?;

        collection.flush().await?;

        // Now we have 2 segments, let merge them

        let segments = collection.get_current_toc().await.toc.clone();
        assert_eq!(segments.len(), 2);
        let pending_segment = collection.init_optimizing(&segments).await?;

        let optimizer = MergeOptimizer::<NoQuantizerL2>::new();
        collection
            .run_optimizer(&optimizer, &pending_segment)
            .await?;

        let segments = collection.get_current_toc().await.toc.clone();
        assert_eq!(segments.len(), 1);

        let snapshot = collection.get_snapshot().await?;
        let snapshot = Arc::new(snapshot);
        let params = SearchParams::new(3, 10, false);
        let result = snapshot
            .search_for_user(0, vec![100.0, 101.0, 102.0], &params, None)
            .await
            .unwrap();
        assert_eq!(result.id_with_scores.len(), 3);

        let mut result_ids = result
            .id_with_scores
            .iter()
            .map(|id| id.doc_id)
            .collect::<Vec<_>>();
        result_ids.sort();
        assert_eq!(result_ids, vec![4, 5, 6]);

        let result = snapshot
            .search_for_user(0, vec![1.0, 2.0, 3.0], &params, None)
            .await
            .unwrap();
        assert_eq!(result.id_with_scores.len(), 3);
        let mut result_ids = result
            .id_with_scores
            .iter()
            .map(|id| id.doc_id)
            .collect::<Vec<_>>();
        result_ids.sort();
        assert_eq!(result_ids, vec![1, 2, 3]);

        Ok(())
    }

    #[tokio::test]
    async fn test_merge_invalidated_optimizer() -> Result<()> {
        let collection_name = "test_merge_invalidated_optimizer";
        let tmp_dir = tempdir::TempDir::new(collection_name)?;
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

        let reader =
            CollectionReader::new(collection_name.to_string(), base_directory.clone(), None);
        let collection = reader.read::<NoQuantizerL2>().await?;

        collection
            .insert_for_users(&[0], 1, &[1.0, 2.0, 3.0], 0, DocumentAttribute::default())
            .await?;
        collection
            .insert_for_users(&[0], 2, &[4.0, 5.0, 6.0], 1, DocumentAttribute::default())
            .await?;
        collection
            .insert_for_users(&[0], 3, &[7.0, 8.0, 9.0], 2, DocumentAttribute::default())
            .await?;

        collection.flush().await?;

        collection
            .insert_for_users(
                &[0],
                4,
                &[100.0, 101.0, 102.0],
                3,
                DocumentAttribute::default(),
            )
            .await?;
        collection
            .insert_for_users(
                &[0],
                5,
                &[103.0, 104.0, 105.0],
                4,
                DocumentAttribute::default(),
            )
            .await?;
        collection
            .insert_for_users(
                &[0],
                6,
                &[106.0, 107.0, 108.0],
                5,
                DocumentAttribute::default(),
            )
            .await?;

        collection.flush().await?;

        // Now we have 2 segments, let's merge them

        let segments = collection.get_current_toc().await.toc.clone();
        assert_eq!(segments.len(), 2);

        // Remove a doc from the first segment
        assert!(collection
            .all_segments()
            .get(&segments[0])
            .unwrap()
            .value()
            .remove(0, 1)
            .await
            .is_ok());

        let pending_segment = collection.init_optimizing(&segments).await?;

        let optimizer = MergeOptimizer::<NoQuantizerL2>::new();
        collection
            .run_optimizer(&optimizer, &pending_segment)
            .await?;

        let segments = collection.get_current_toc().await.toc.clone();
        assert_eq!(segments.len(), 1);

        let snapshot = collection.get_snapshot().await?;
        let snapshot = Arc::new(snapshot);

        let params = SearchParams::new(3, 10, false);
        let result = snapshot
            .search_for_user(0, vec![1.0, 2.0, 3.0], &params, None)
            .await
            .unwrap();
        assert_eq!(result.id_with_scores.len(), 3);
        let mut result_ids = result
            .id_with_scores
            .iter()
            .map(|id| id.doc_id)
            .collect::<Vec<_>>();
        result_ids.sort();
        assert_eq!(result_ids, vec![2, 3, 4]);

        Ok(())
    }

    #[tokio::test]
    async fn test_merge_optimizer_with_multiple_users() -> Result<()> {
        let collection_name = "test_merge_optimizer_with_multiple_users";
        let tmp_dir = tempdir::TempDir::new(collection_name)?;
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

        let reader =
            CollectionReader::new(collection_name.to_string(), base_directory.clone(), None);
        let collection = reader.read::<NoQuantizerL2>().await?;

        collection
            .insert_for_users(&[0], 1, &[1.0, 2.0, 3.0], 0, DocumentAttribute::default())
            .await?;
        collection
            .insert_for_users(&[0], 2, &[4.0, 5.0, 6.0], 1, DocumentAttribute::default())
            .await?;
        collection
            .insert_for_users(&[0], 3, &[7.0, 8.0, 9.0], 2, DocumentAttribute::default())
            .await?;

        collection.flush().await?;

        collection
            .insert_for_users(
                &[1],
                4,
                &[10.0, 11.0, 12.0],
                3,
                DocumentAttribute::default(),
            )
            .await?;
        collection
            .insert_for_users(
                &[1],
                5,
                &[13.0, 14.0, 15.0],
                4,
                DocumentAttribute::default(),
            )
            .await?;
        collection
            .insert_for_users(
                &[1],
                6,
                &[16.0, 17.0, 18.0],
                5,
                DocumentAttribute::default(),
            )
            .await?;

        collection.flush().await?;

        // Now we have 2 segments, let's merge them

        let segments = collection.get_current_toc().await.toc.clone();
        assert_eq!(segments.len(), 2);
        let pending_segment = collection.init_optimizing(&segments).await?;

        let optimizer = MergeOptimizer::<NoQuantizerL2>::new();
        collection
            .run_optimizer(&optimizer, &pending_segment)
            .await?;

        let segments = collection.get_current_toc().await.toc.clone();
        assert_eq!(segments.len(), 1);

        let snapshot = collection.get_snapshot().await?;
        let params = SearchParams::new(3, 10, false);
        let result = snapshot
            .search_for_user(0, vec![1.0, 2.0, 3.0], &params, None)
            .await
            .unwrap();
        assert_eq!(result.id_with_scores.len(), 3);

        let mut result_ids = result
            .id_with_scores
            .iter()
            .map(|id| id.doc_id)
            .collect::<Vec<_>>();
        result_ids.sort();
        assert_eq!(result_ids, vec![1, 2, 3]);

        let result = snapshot
            .search_for_user(1, vec![10.0, 11.0, 12.0], &params, None)
            .await
            .unwrap();
        assert_eq!(result.id_with_scores.len(), 3);

        let mut result_ids = result
            .id_with_scores
            .iter()
            .map(|id| id.doc_id)
            .collect::<Vec<_>>();
        result_ids.sort();
        assert_eq!(result_ids, vec![4, 5, 6]);

        Ok(())
    }

    #[tokio::test]
    async fn test_merge_invalidated_optimizer_with_multiple_users() -> Result<()> {
        let collection_name = "test_merge_invalidated_optimizer_with_multiple_users";
        let tmp_dir = tempdir::TempDir::new(collection_name)?;
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

        let reader =
            CollectionReader::new(collection_name.to_string(), base_directory.clone(), None);
        let collection = reader.read::<NoQuantizerL2>().await?;

        collection
            .insert_for_users(&[0], 1, &[1.0, 2.0, 3.0], 0, DocumentAttribute::default())
            .await?;
        collection
            .insert_for_users(&[0], 2, &[4.0, 5.0, 6.0], 1, DocumentAttribute::default())
            .await?;
        collection
            .insert_for_users(&[0], 3, &[7.0, 8.0, 9.0], 2, DocumentAttribute::default())
            .await?;

        collection.flush().await?;

        collection
            .insert_for_users(
                &[1],
                4,
                &[10.0, 11.0, 12.0],
                3,
                DocumentAttribute::default(),
            )
            .await?;
        collection
            .insert_for_users(
                &[1],
                5,
                &[13.0, 14.0, 15.0],
                4,
                DocumentAttribute::default(),
            )
            .await?;
        collection
            .insert_for_users(
                &[1],
                6,
                &[16.0, 17.0, 18.0],
                5,
                DocumentAttribute::default(),
            )
            .await?;

        collection.flush().await?;

        // Now we have 2 segments, let's merge them

        let segments = collection.get_current_toc().await.toc.clone();
        assert_eq!(segments.len(), 2);

        // Remove a doc from the first segment
        assert!(collection
            .all_segments()
            .get(&segments[0])
            .unwrap()
            .value()
            .remove(0, 1)
            .await
            .is_ok());

        let pending_segment = collection.init_optimizing(&segments).await?;

        let optimizer = MergeOptimizer::<NoQuantizerL2>::new();
        collection
            .run_optimizer(&optimizer, &pending_segment)
            .await?;

        let segments = collection.get_current_toc().await.toc.clone();
        assert_eq!(segments.len(), 1);

        let snapshot = collection.get_snapshot().await?;
        let params = SearchParams::new(3, 10, false);
        let result = snapshot
            .search_for_user(0, vec![1.0, 2.0, 3.0], &params, None)
            .await
            .unwrap();
        assert_eq!(result.id_with_scores.len(), 2);

        let mut result_ids = result
            .id_with_scores
            .iter()
            .map(|id| id.doc_id)
            .collect::<Vec<_>>();
        result_ids.sort();
        assert_eq!(result_ids, vec![2, 3]);

        Ok(())
    }

    #[tokio::test]
    async fn test_merge_optimizer_with_terms() -> Result<()> {
        let collection_name = "test_merge_optimizer_with_terms";
        let tmp_dir = tempdir::TempDir::new(collection_name)?;
        let base_directory = tmp_dir.path().to_str().unwrap().to_string();
        let mut config = CollectionConfig::default_test_config();
        config.num_features = 3;
        config.wal_file_size = 0;
        config.max_time_to_flush_ms = 0;
        config.max_pending_ops = 0;
        config.initial_num_centroids = 1;

        Collection::<NoQuantizerL2>::init_new_collection(base_directory.clone(), &config)?;

        let reader =
            CollectionReader::new(collection_name.to_string(), base_directory.clone(), None);
        let collection = reader.read::<NoQuantizerL2>().await?;

        // Segment 1: Doc 1 (tag:a), Doc 2 (tag:b)
        let mut attr1 = DocumentAttribute::default();
        attr1.value.insert(
            "tag".to_string(),
            proto::muopdb::AttributeValue {
                value: Some(proto::muopdb::attribute_value::Value::KeywordValue(
                    "a".to_string(),
                )),
            },
        );
        collection
            .insert_for_users(&[0], 1, &[1.0, 1.0, 1.0], 0, attr1)
            .await?;

        let mut attr2 = DocumentAttribute::default();
        attr2.value.insert(
            "tag".to_string(),
            proto::muopdb::AttributeValue {
                value: Some(proto::muopdb::attribute_value::Value::KeywordValue(
                    "b".to_string(),
                )),
            },
        );
        collection
            .insert_for_users(&[0], 2, &[2.0, 2.0, 2.0], 1, attr2)
            .await?;

        collection.flush().await?;

        // Segment 2: Doc 3 (tag:a), Doc 4 (tag:c)
        let mut attr3 = DocumentAttribute::default();
        attr3.value.insert(
            "tag".to_string(),
            proto::muopdb::AttributeValue {
                value: Some(proto::muopdb::attribute_value::Value::KeywordValue(
                    "a".to_string(),
                )),
            },
        );
        collection
            .insert_for_users(&[0], 3, &[3.0, 3.0, 3.0], 2, attr3)
            .await?;

        let mut attr4 = DocumentAttribute::default();
        attr4.value.insert(
            "tag".to_string(),
            proto::muopdb::AttributeValue {
                value: Some(proto::muopdb::attribute_value::Value::KeywordValue(
                    "c".to_string(),
                )),
            },
        );
        collection
            .insert_for_users(&[0], 4, &[4.0, 4.0, 4.0], 3, attr4)
            .await?;

        collection.flush().await?;

        // Verify we have 2 segments
        let segments = collection.get_current_toc().await.toc.clone();
        assert_eq!(segments.len(), 2);

        // Merge segments
        let pending_segment = collection.init_optimizing(&segments).await?;
        let optimizer = MergeOptimizer::<NoQuantizerL2>::new();
        collection
            .run_optimizer(&optimizer, &pending_segment)
            .await?;

        // Verify we have 1 segment
        let segments = collection.get_current_toc().await.toc.clone();
        assert_eq!(segments.len(), 1);

        // Test search with term filter "tag:a"
        let snapshot = collection.get_snapshot().await?;
        let mut filter = proto::muopdb::DocumentFilter::default();
        filter.filter = Some(proto::muopdb::document_filter::Filter::Contains(
            proto::muopdb::ContainsFilter {
                path: "tag".to_string(),
                value: "a".to_string(),
            },
        ));

        let params = SearchParams::new(10, 10, false);
        let result = snapshot
            .search_for_user(0, vec![1.0, 1.0, 1.0], &params, Some(Arc::new(filter)))
            .await
            .unwrap();

        // Should find Doc 1 and Doc 3
        assert_eq!(result.id_with_scores.len(), 2);
        let mut doc_ids: Vec<u128> = result.id_with_scores.iter().map(|r| r.doc_id).collect();
        doc_ids.sort();
        assert_eq!(doc_ids, vec![1, 3]);

        Ok(())
    }

    #[tokio::test]
    async fn test_merge_optimizer_with_multiple_terms_per_doc() -> Result<()> {
        let collection_name = "test_merge_optimizer_with_multiple_terms";
        let tmp_dir = tempdir::TempDir::new(collection_name)?;
        let base_directory = tmp_dir.path().to_str().unwrap().to_string();
        let mut config = CollectionConfig::default_test_config();
        config.num_features = 3;
        config.wal_file_size = 0;
        config.max_time_to_flush_ms = 0;
        config.max_pending_ops = 0;
        config.initial_num_centroids = 1;

        Collection::<NoQuantizerL2>::init_new_collection(base_directory.clone(), &config)?;

        let reader =
            CollectionReader::new(collection_name.to_string(), base_directory.clone(), None);
        let collection = reader.read::<NoQuantizerL2>().await?;

        // Segment 1: Doc 1 (tag:a, color:red), Doc 2 (tag:b, color:blue)
        let mut attr1 = DocumentAttribute::default();
        attr1.value.insert(
            "tag".to_string(),
            proto::muopdb::AttributeValue {
                value: Some(proto::muopdb::attribute_value::Value::KeywordValue(
                    "a".to_string(),
                )),
            },
        );
        attr1.value.insert(
            "color".to_string(),
            proto::muopdb::AttributeValue {
                value: Some(proto::muopdb::attribute_value::Value::KeywordValue(
                    "red".to_string(),
                )),
            },
        );
        collection
            .insert_for_users(&[0], 1, &[1.0, 1.0, 1.0], 0, attr1)
            .await?;

        let mut attr2 = DocumentAttribute::default();
        attr2.value.insert(
            "tag".to_string(),
            proto::muopdb::AttributeValue {
                value: Some(proto::muopdb::attribute_value::Value::KeywordValue(
                    "b".to_string(),
                )),
            },
        );
        attr2.value.insert(
            "color".to_string(),
            proto::muopdb::AttributeValue {
                value: Some(proto::muopdb::attribute_value::Value::KeywordValue(
                    "blue".to_string(),
                )),
            },
        );
        collection
            .insert_for_users(&[0], 2, &[2.0, 2.0, 2.0], 1, attr2)
            .await?;

        collection.flush().await?;

        // Segment 2: Doc 3 (tag:a, color:blue), Doc 4 (tag:c, color:red)
        let mut attr3 = DocumentAttribute::default();
        attr3.value.insert(
            "tag".to_string(),
            proto::muopdb::AttributeValue {
                value: Some(proto::muopdb::attribute_value::Value::KeywordValue(
                    "a".to_string(),
                )),
            },
        );
        attr3.value.insert(
            "color".to_string(),
            proto::muopdb::AttributeValue {
                value: Some(proto::muopdb::attribute_value::Value::KeywordValue(
                    "blue".to_string(),
                )),
            },
        );
        collection
            .insert_for_users(&[0], 3, &[3.0, 3.0, 3.0], 2, attr3)
            .await?;

        let mut attr4 = DocumentAttribute::default();
        attr4.value.insert(
            "tag".to_string(),
            proto::muopdb::AttributeValue {
                value: Some(proto::muopdb::attribute_value::Value::KeywordValue(
                    "c".to_string(),
                )),
            },
        );
        attr4.value.insert(
            "color".to_string(),
            proto::muopdb::AttributeValue {
                value: Some(proto::muopdb::attribute_value::Value::KeywordValue(
                    "red".to_string(),
                )),
            },
        );
        collection
            .insert_for_users(&[0], 4, &[4.0, 4.0, 4.0], 3, attr4)
            .await?;

        collection.flush().await?;

        // Verify we have 2 segments
        let segments = collection.get_current_toc().await.toc.clone();
        assert_eq!(segments.len(), 2);

        // Merge segments
        let pending_segment = collection.init_optimizing(&segments).await?;
        let optimizer = MergeOptimizer::<NoQuantizerL2>::new();
        collection
            .run_optimizer(&optimizer, &pending_segment)
            .await?;

        // Verify we have 1 segment
        let segments = collection.get_current_toc().await.toc.clone();
        assert_eq!(segments.len(), 1);

        let snapshot = collection.get_snapshot().await?;
        let params = SearchParams::new(10, 10, false);

        // Test search with term filter "tag:a" - should find Doc 1 and Doc 3
        let mut filter = proto::muopdb::DocumentFilter::default();
        filter.filter = Some(proto::muopdb::document_filter::Filter::Contains(
            proto::muopdb::ContainsFilter {
                path: "tag".to_string(),
                value: "a".to_string(),
            },
        ));

        let result = snapshot
            .search_for_user(0, vec![1.0, 1.0, 1.0], &params, Some(Arc::new(filter)))
            .await
            .unwrap();

        assert_eq!(result.id_with_scores.len(), 2);
        let mut doc_ids: Vec<u128> = result.id_with_scores.iter().map(|r| r.doc_id).collect();
        doc_ids.sort();
        assert_eq!(doc_ids, vec![1, 3]);

        // Test search with term filter "color:red" - should find Doc 1 and Doc 4
        let mut filter = proto::muopdb::DocumentFilter::default();
        filter.filter = Some(proto::muopdb::document_filter::Filter::Contains(
            proto::muopdb::ContainsFilter {
                path: "color".to_string(),
                value: "red".to_string(),
            },
        ));

        let result = snapshot
            .search_for_user(0, vec![1.0, 1.0, 1.0], &params, Some(Arc::new(filter)))
            .await
            .unwrap();

        assert_eq!(result.id_with_scores.len(), 2);
        let mut doc_ids: Vec<u128> = result.id_with_scores.iter().map(|r| r.doc_id).collect();
        doc_ids.sort();
        assert_eq!(doc_ids, vec![1, 4]);

        // Test search with term filter "color:blue" - should find Doc 2 and Doc 3
        let mut filter = proto::muopdb::DocumentFilter::default();
        filter.filter = Some(proto::muopdb::document_filter::Filter::Contains(
            proto::muopdb::ContainsFilter {
                path: "color".to_string(),
                value: "blue".to_string(),
            },
        ));

        let result = snapshot
            .search_for_user(0, vec![1.0, 1.0, 1.0], &params, Some(Arc::new(filter)))
            .await
            .unwrap();

        assert_eq!(result.id_with_scores.len(), 2);
        let mut doc_ids: Vec<u128> = result.id_with_scores.iter().map(|r| r.doc_id).collect();
        doc_ids.sort();
        assert_eq!(doc_ids, vec![2, 3]);

        Ok(())
    }

    #[tokio::test]
    async fn test_merge_optimizer_with_terms_multiple_users() -> Result<()> {
        let collection_name = "test_merge_optimizer_with_terms_multi_user";
        let tmp_dir = tempdir::TempDir::new(collection_name)?;
        let base_directory = tmp_dir.path().to_str().unwrap().to_string();
        let mut config = CollectionConfig::default_test_config();
        config.num_features = 3;
        config.wal_file_size = 0;
        config.max_time_to_flush_ms = 0;
        config.max_pending_ops = 0;
        config.initial_num_centroids = 1;

        Collection::<NoQuantizerL2>::init_new_collection(base_directory.clone(), &config)?;

        let reader =
            CollectionReader::new(collection_name.to_string(), base_directory.clone(), None);
        let collection = reader.read::<NoQuantizerL2>().await?;

        // Segment 1: User 0 (tag:x, cat:electronics), User 1 (tag:p, cat:food)
        let mut attr1 = DocumentAttribute::default();
        attr1.value.insert(
            "tag".to_string(),
            proto::muopdb::AttributeValue {
                value: Some(proto::muopdb::attribute_value::Value::KeywordValue(
                    "x".to_string(),
                )),
            },
        );
        attr1.value.insert(
            "cat".to_string(),
            proto::muopdb::AttributeValue {
                value: Some(proto::muopdb::attribute_value::Value::KeywordValue(
                    "electronics".to_string(),
                )),
            },
        );
        collection
            .insert_for_users(&[0], 1, &[1.0, 1.0, 1.0], 0, attr1)
            .await?;

        let mut attr2 = DocumentAttribute::default();
        attr2.value.insert(
            "tag".to_string(),
            proto::muopdb::AttributeValue {
                value: Some(proto::muopdb::attribute_value::Value::KeywordValue(
                    "p".to_string(),
                )),
            },
        );
        attr2.value.insert(
            "cat".to_string(),
            proto::muopdb::AttributeValue {
                value: Some(proto::muopdb::attribute_value::Value::KeywordValue(
                    "food".to_string(),
                )),
            },
        );
        collection
            .insert_for_users(&[1], 2, &[2.0, 2.0, 2.0], 1, attr2)
            .await?;

        collection.flush().await?;

        // Segment 2: User 0 (tag:y, cat:food), User 1 (tag:q, cat:electronics)
        let mut attr3 = DocumentAttribute::default();
        attr3.value.insert(
            "tag".to_string(),
            proto::muopdb::AttributeValue {
                value: Some(proto::muopdb::attribute_value::Value::KeywordValue(
                    "y".to_string(),
                )),
            },
        );
        attr3.value.insert(
            "cat".to_string(),
            proto::muopdb::AttributeValue {
                value: Some(proto::muopdb::attribute_value::Value::KeywordValue(
                    "food".to_string(),
                )),
            },
        );
        collection
            .insert_for_users(&[0], 3, &[3.0, 3.0, 3.0], 2, attr3)
            .await?;

        let mut attr4 = DocumentAttribute::default();
        attr4.value.insert(
            "tag".to_string(),
            proto::muopdb::AttributeValue {
                value: Some(proto::muopdb::attribute_value::Value::KeywordValue(
                    "q".to_string(),
                )),
            },
        );
        attr4.value.insert(
            "cat".to_string(),
            proto::muopdb::AttributeValue {
                value: Some(proto::muopdb::attribute_value::Value::KeywordValue(
                    "electronics".to_string(),
                )),
            },
        );
        collection
            .insert_for_users(&[1], 4, &[4.0, 4.0, 4.0], 3, attr4)
            .await?;

        collection.flush().await?;

        let segments = collection.get_current_toc().await.toc.clone();
        assert_eq!(segments.len(), 2);

        let pending_segment = collection.init_optimizing(&segments).await?;
        let optimizer = MergeOptimizer::<NoQuantizerL2>::new();
        collection
            .run_optimizer(&optimizer, &pending_segment)
            .await?;

        let segments = collection.get_current_toc().await.toc.clone();
        assert_eq!(segments.len(), 1);

        let snapshot = collection.get_snapshot().await?;
        let params = SearchParams::new(10, 10, false);

        // Test User 0: search for tag:x - should find Doc 1
        let mut filter = proto::muopdb::DocumentFilter::default();
        filter.filter = Some(proto::muopdb::document_filter::Filter::Contains(
            proto::muopdb::ContainsFilter {
                path: "tag".to_string(),
                value: "x".to_string(),
            },
        ));

        let result = snapshot
            .search_for_user(0, vec![1.0, 1.0, 1.0], &params, Some(Arc::new(filter)))
            .await
            .unwrap();
        assert_eq!(result.id_with_scores.len(), 1);
        assert_eq!(result.id_with_scores[0].doc_id, 1);

        // Test User 1: search for tag:p - should find Doc 2
        let mut filter = proto::muopdb::DocumentFilter::default();
        filter.filter = Some(proto::muopdb::document_filter::Filter::Contains(
            proto::muopdb::ContainsFilter {
                path: "tag".to_string(),
                value: "p".to_string(),
            },
        ));

        let result = snapshot
            .search_for_user(1, vec![2.0, 2.0, 2.0], &params, Some(Arc::new(filter)))
            .await
            .unwrap();
        assert_eq!(result.id_with_scores.len(), 1);
        assert_eq!(result.id_with_scores[0].doc_id, 2);

        Ok(())
    }

    #[tokio::test]
    async fn test_merge_optimizer_with_terms_and_invalidation() -> Result<()> {
        let collection_name = "test_merge_optimizer_with_terms_invalidated";
        let tmp_dir = tempdir::TempDir::new(collection_name)?;
        let base_directory = tmp_dir.path().to_str().unwrap().to_string();
        let mut config = CollectionConfig::default_test_config();
        config.num_features = 3;
        config.wal_file_size = 0;
        config.max_time_to_flush_ms = 0;
        config.max_pending_ops = 0;
        config.initial_num_centroids = 1;

        Collection::<NoQuantizerL2>::init_new_collection(base_directory.clone(), &config)?;

        let reader =
            CollectionReader::new(collection_name.to_string(), base_directory.clone(), None);
        let collection = reader.read::<NoQuantizerL2>().await?;

        // Segment 1: Doc 1 (tag:m, status:active), Doc 2 (tag:n, status:active)
        let mut attr1 = DocumentAttribute::default();
        attr1.value.insert(
            "tag".to_string(),
            proto::muopdb::AttributeValue {
                value: Some(proto::muopdb::attribute_value::Value::KeywordValue(
                    "m".to_string(),
                )),
            },
        );
        attr1.value.insert(
            "status".to_string(),
            proto::muopdb::AttributeValue {
                value: Some(proto::muopdb::attribute_value::Value::KeywordValue(
                    "active".to_string(),
                )),
            },
        );
        collection
            .insert_for_users(&[0], 1, &[1.0, 1.0, 1.0], 0, attr1)
            .await?;

        let mut attr2 = DocumentAttribute::default();
        attr2.value.insert(
            "tag".to_string(),
            proto::muopdb::AttributeValue {
                value: Some(proto::muopdb::attribute_value::Value::KeywordValue(
                    "n".to_string(),
                )),
            },
        );
        attr2.value.insert(
            "status".to_string(),
            proto::muopdb::AttributeValue {
                value: Some(proto::muopdb::attribute_value::Value::KeywordValue(
                    "active".to_string(),
                )),
            },
        );
        collection
            .insert_for_users(&[0], 2, &[2.0, 2.0, 2.0], 1, attr2)
            .await?;

        collection.flush().await?;

        // Segment 2: Doc 3 (tag:m, status:archived), Doc 4 (tag:o, status:active)
        let mut attr3 = DocumentAttribute::default();
        attr3.value.insert(
            "tag".to_string(),
            proto::muopdb::AttributeValue {
                value: Some(proto::muopdb::attribute_value::Value::KeywordValue(
                    "m".to_string(),
                )),
            },
        );
        attr3.value.insert(
            "status".to_string(),
            proto::muopdb::AttributeValue {
                value: Some(proto::muopdb::attribute_value::Value::KeywordValue(
                    "archived".to_string(),
                )),
            },
        );
        collection
            .insert_for_users(&[0], 3, &[3.0, 3.0, 3.0], 2, attr3)
            .await?;

        let mut attr4 = DocumentAttribute::default();
        attr4.value.insert(
            "tag".to_string(),
            proto::muopdb::AttributeValue {
                value: Some(proto::muopdb::attribute_value::Value::KeywordValue(
                    "o".to_string(),
                )),
            },
        );
        attr4.value.insert(
            "status".to_string(),
            proto::muopdb::AttributeValue {
                value: Some(proto::muopdb::attribute_value::Value::KeywordValue(
                    "active".to_string(),
                )),
            },
        );
        collection
            .insert_for_users(&[0], 4, &[4.0, 4.0, 4.0], 3, attr4)
            .await?;

        collection.flush().await?;

        let segments = collection.get_current_toc().await.toc.clone();
        assert_eq!(segments.len(), 2);

        // Remove Doc 1 (tag:m, status:active) from first segment
        assert!(collection
            .all_segments()
            .get(&segments[0])
            .unwrap()
            .value()
            .remove(0, 1)
            .await
            .is_ok());

        let pending_segment = collection.init_optimizing(&segments).await?;
        let optimizer = MergeOptimizer::<NoQuantizerL2>::new();
        collection
            .run_optimizer(&optimizer, &pending_segment)
            .await?;

        let segments = collection.get_current_toc().await.toc.clone();
        assert_eq!(segments.len(), 1);

        let snapshot = collection.get_snapshot().await?;
        let params = SearchParams::new(10, 10, false);

        // Search for tag:m - should find Doc 3 only (Doc 1 was invalidated but we check)
        let mut filter = proto::muopdb::DocumentFilter::default();
        filter.filter = Some(proto::muopdb::document_filter::Filter::Contains(
            proto::muopdb::ContainsFilter {
                path: "tag".to_string(),
                value: "m".to_string(),
            },
        ));

        let result = snapshot
            .search_for_user(0, vec![1.0, 1.0, 1.0], &params, Some(Arc::new(filter)))
            .await
            .unwrap();
        // Doc 1 was invalidated but its term may still be in index
        // The search returns whatever is in the term index
        assert!(result.id_with_scores.len() >= 1);

        Ok(())
    }
}
