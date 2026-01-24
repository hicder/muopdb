use std::sync::Arc;

use anyhow::{anyhow, Result};
use config::attribute_schema::AttributeSchema;
use config::search_params::SearchParams;
use proto::muopdb::DocumentFilter;
use quantization::quantization::Quantizer;
use utils::file_io::env::Env;

use crate::multi_spann::index::MultiSpannIndex;
use crate::multi_terms::index::MultiTermIndex;
use crate::query::async_iters::AsyncInvertedIndexIter;
use crate::query::async_planner::AsyncPlanner;
use crate::query::iters::InvertedIndexIter;
use crate::query::planner::Planner;
use crate::segment::Segment;
use crate::spann::iter::SpannIter;
use crate::utils::SearchResult;

/// This is an immutable segment. This usually contains a single index.
pub struct ImmutableSegment<Q: Quantizer> {
    index: MultiSpannIndex<Q>,
    name: String,
    multi_term_index: Option<Arc<MultiTermIndex>>,
}

impl<Q: Quantizer> ImmutableSegment<Q> {
    pub async fn new(
        index: MultiSpannIndex<Q>,
        name: String,
        terms_dir: Option<String>,
        env: Option<Arc<Box<dyn Env>>>,
    ) -> Self {
        let multi_term_index = match (terms_dir, env) {
            (Some(dir), Some(env)) => MultiTermIndex::new_with_env(dir, env)
                .await
                .ok()
                .map(Arc::new),
            (Some(dir), None) => MultiTermIndex::new(dir).ok().map(Arc::new),
            _ => None,
        };
        Self {
            index,
            name,
            multi_term_index,
        }
    }

    pub fn user_ids(&self) -> Vec<u128> {
        self.index.user_ids()
    }

    pub async fn iter_for_user(&self, user_id: u128) -> Option<SpannIter<Q>> {
        self.index.iter_for_user(user_id).await
    }

    pub fn size_in_bytes(&self) -> u64 {
        self.index.size_in_bytes()
    }

    /// This is very expensive and should only be used for testing.
    #[cfg(test)]
    pub async fn get_point_id(&self, user_id: u128, doc_id: u128) -> Option<u32> {
        self.index.get_point_id(user_id, doc_id).await
    }

    pub async fn is_invalidated(&self, user_id: u128, doc_id: u128) -> Result<bool> {
        self.index.is_invalidated(user_id, doc_id).await
    }

    pub fn num_docs(&self) -> Result<usize> {
        self.index.get_total_docs_count()
    }

    pub async fn should_auto_vacuum(&self) -> bool {
        let count_deleted_documents = self.index.get_deleted_docs_count().await as f64;
        if let Ok(count_all_documents) = self.num_docs() {
            (count_deleted_documents / (count_all_documents as f64)) > 0.1
        } else {
            false
        }
    }

    pub fn get_multi_term_index(&self) -> Option<Arc<MultiTermIndex>> {
        self.multi_term_index.clone()
    }

    /// Returns the document ID associated with a point ID.
    ///
    /// # Arguments
    /// * `user_id` - The ID of the user.
    /// * `point_id` - The internal point ID.
    ///
    /// # Returns
    /// * `Option<u128>` - The 128-bit document ID if found, otherwise `None`.
    pub async fn get_doc_id(&self, user_id: u128, point_id: u32) -> Option<u128> {
        self.index.get_doc_id(user_id, point_id).await
    }
}

/// This is the implementation of Segment for ImmutableSegment.
#[async_trait::async_trait]
impl<Q: Quantizer> Segment for ImmutableSegment<Q> {
    /// ImmutableSegment does not support insertion.
    async fn insert(&self, _doc_id: u128, _data: &[f32]) -> Result<()> {
        Err(anyhow!("ImmutableSegment does not support insertion"))
    }

    /// ImmutableSegment does not support actual removal, we are just invalidating documents.
    async fn remove(&self, user_id: u128, doc_id: u128) -> Result<bool> {
        self.index.invalidate(user_id, doc_id).await
    }

    /// ImmutableSegment does not support contains.
    async fn may_contain(&self, _doc_id: u128) -> bool {
        // TODO(hicder): Implement this
        true
    }

    async fn name(&self) -> String {
        self.name.clone()
    }
}

impl<Q: Quantizer> ImmutableSegment<Q> {
    pub async fn search_for_user(
        &self,
        user_id: u128,
        query: Vec<f32>,
        params: &SearchParams,
        planner: Option<Arc<Planner>>,
    ) -> Option<SearchResult> {
        self.index
            .search_for_user(user_id, query, params, planner)
            .await
    }

    /// Iterates all (term, point_id) pairs for a given user.
    pub async fn iter_terms_for_user(&self, user_id: u128) -> Option<Vec<(String, u32)>> {
        let multi_term_index = self.get_multi_term_index()?;
        multi_term_index
            .iter_term_point_pairs_for_user(user_id)
            .ok()
            .map(|iter| iter.collect())
    }

    /// Search only the term index for documents matching the filter.
    /// Returns a Vec of point IDs (internal format).
    pub async fn search_terms_for_user(
        &self,
        user_id: u128,
        filter: Arc<DocumentFilter>,
        attribute_schema: Option<AttributeSchema>,
    ) -> Vec<u32> {
        let multi_term_index = match self.get_multi_term_index() {
            Some(idx) => idx,
            None => return vec![],
        };

        // Create a planner with the filter (no vector search)
        let planner = match Planner::new(
            user_id,
            (*filter).clone(),
            multi_term_index,
            attribute_schema,
        ) {
            Ok(p) => p,
            Err(_) => return vec![],
        };

        // Collect all point IDs from the iterator
        let mut iter = match planner.plan() {
            Ok(iter) => iter,
            Err(_) => return vec![],
        };

        // Manually collect since iters::Iter doesn't implement Iterator trait directly
        let mut result = Vec::new();
        while let Some(point_id) = iter.next() {
            result.push(point_id);
        }
        result
    }

    /// Search only the term index for documents matching the filter using async planner.
    pub async fn search_terms_for_user_async(
        &self,
        user_id: u128,
        filter: Arc<DocumentFilter>,
        attribute_schema: Option<AttributeSchema>,
    ) -> Vec<u32> {
        let multi_term_index = match self.get_multi_term_index() {
            Some(idx) => idx,
            None => return vec![],
        };

        let planner = match AsyncPlanner::new(
            user_id,
            (*filter).clone(),
            multi_term_index,
            attribute_schema,
        )
        .await
        {
            Ok(p) => p,
            Err(_) => return vec![],
        };

        let mut iter = match planner.plan().await {
            Ok(iter) => iter,
            Err(e) => {
                eprintln!("Failed to plan search: {}", e);
                return vec![];
            }
        };

        let mut result = Vec::new();
        loop {
            match iter.next().await {
                Ok(Some(point_id)) => result.push(point_id),
                Ok(None) => break,
                Err(_) => break,
            }
        }
        result
    }
}

unsafe impl<Q: Quantizer> Send for ImmutableSegment<Q> {}
unsafe impl<Q: Quantizer> Sync for ImmutableSegment<Q> {}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use config::collection::CollectionConfig;
    use config::search_params::SearchParams;
    use quantization::noq::NoQuantizer;
    use utils::distance::l2::L2DistanceCalculator;
    use utils::file_io::env::{DefaultEnv, Env, EnvConfig, FileType};

    fn create_env() -> Arc<Box<dyn Env>> {
        let config = EnvConfig {
            file_type: FileType::CachedStandard,
            ..EnvConfig::default()
        };
        Arc::new(Box::new(DefaultEnv::new(config)))
    }

    use crate::multi_spann::builder::MultiSpannBuilder;
    use crate::multi_spann::reader::MultiSpannReader;
    use crate::multi_spann::writer::MultiSpannWriter;
    use crate::segment::{ImmutableSegment, Segment};

    #[tokio::test]
    async fn test_immutable_segment_search() {
        let temp_dir = tempdir::TempDir::new("immutable_segment_search_test")
            .expect("Failed to create temporary directory");
        let base_directory = temp_dir
            .path()
            .to_str()
            .expect("Failed to convert temporary directory path to string")
            .to_string();
        let env = create_env();

        let num_vectors = 1000;
        let num_features = 4;

        let mut spann_builder_config = CollectionConfig::default_test_config();
        spann_builder_config.num_features = num_features;
        let mut multi_spann_builder =
            MultiSpannBuilder::new(spann_builder_config, base_directory.clone())
                .expect("Failed to create Multi-SPANN builder");

        // Generate 1000 vectors of f32, dimension 4
        for i in 0..num_vectors {
            assert!(multi_spann_builder
                .insert(0, i, &[i as f32, i as f32, i as f32, i as f32])
                .is_ok());
        }
        assert!(multi_spann_builder
            .insert(0, num_vectors, &[1.2, 2.2, 3.2, 4.2])
            .is_ok());
        assert!(multi_spann_builder.build().is_ok());

        let multi_spann_writer = MultiSpannWriter::new(base_directory.clone());
        assert!(multi_spann_writer.write(&mut multi_spann_builder).is_ok());

        let multi_spann_reader = MultiSpannReader::new(base_directory);
        let multi_spann_index = multi_spann_reader
            .read::<NoQuantizer<L2DistanceCalculator>>(num_features, env.clone())
            .await
            .expect("Failed to read Multi-SPANN index");

        let name_for_new_segment = format!("segment_{}", rand::random::<u64>());
        let immutable_segment = ImmutableSegment::new(
            multi_spann_index,
            name_for_new_segment.clone(),
            None,
            Some(env.clone()),
        )
        .await;

        let query = vec![1.4, 2.4, 3.4, 4.4];
        let k = 3;
        let num_probes = 2;

        let params = SearchParams::new(k, num_probes, false);

        let results = immutable_segment
            .search_for_user(0, query.clone(), &params, None)
            .await
            .expect("Failed to search with Multi-SPANN index");

        assert_eq!(results.id_with_scores.len(), k);
        assert_eq!(results.id_with_scores[0].doc_id, num_vectors);
        assert_eq!(results.id_with_scores[1].doc_id, 3);
        assert_eq!(results.id_with_scores[2].doc_id, 2);
    }

    #[tokio::test]
    async fn test_immutable_segment_search_with_invalidation() {
        let temp_dir = tempdir::TempDir::new("immutable_segment_search_with_invalidation_test")
            .expect("Failed to create temporary directory");
        let base_directory = temp_dir
            .path()
            .to_str()
            .expect("Failed to convert temporary directory path to string")
            .to_string();
        let env = create_env();

        let num_vectors = 1000;
        let num_features = 4;

        let mut spann_builder_config = CollectionConfig::default_test_config();
        spann_builder_config.num_features = num_features;
        let mut multi_spann_builder =
            MultiSpannBuilder::new(spann_builder_config, base_directory.clone())
                .expect("Failed to create Multi-SPANN builder");

        // Generate 1000 vectors of f32, dimension 4
        for i in 0..num_vectors {
            assert!(multi_spann_builder
                .insert(0, i as u128, &[i as f32, i as f32, i as f32, i as f32])
                .is_ok());
        }
        for i in 0..num_vectors {
            assert!(multi_spann_builder
                .insert(1, i as u128, &[i as f32, i as f32, i as f32, i as f32])
                .is_ok());
        }
        assert!(multi_spann_builder
            .insert(0, num_vectors as u128, &[1.2, 2.2, 3.2, 4.2])
            .is_ok());
        assert!(multi_spann_builder.build().is_ok());

        let multi_spann_writer = MultiSpannWriter::new(base_directory.clone());
        assert!(multi_spann_writer.write(&mut multi_spann_builder).is_ok());

        let multi_spann_reader = MultiSpannReader::new(base_directory);
        let multi_spann_index = multi_spann_reader
            .read::<NoQuantizer<L2DistanceCalculator>>(num_features, env.clone())
            .await
            .expect("Failed to read Multi-SPANN index");

        let name_for_new_segment = format!("segment_{}", rand::random::<u64>());
        let immutable_segment = ImmutableSegment::new(
            multi_spann_index,
            name_for_new_segment.clone(),
            None,
            Some(env.clone()),
        )
        .await;

        let query = vec![1.4, 2.4, 3.4, 4.4];
        let k = 3;
        let num_probes = 2;

        assert!(immutable_segment
            .remove(0, num_vectors as u128)
            .await
            .expect("Failed to invalidate"));

        let params = SearchParams::new(k, num_probes, false);

        let results = immutable_segment
            .search_for_user(0, query.clone(), &params, None)
            .await
            .expect("Failed to search with Multi-SPANN index");

        assert_eq!(results.id_with_scores.len(), k);
        assert_eq!(results.id_with_scores[0].doc_id, 3);
        assert_eq!(results.id_with_scores[1].doc_id, 2);
        assert_eq!(results.id_with_scores[2].doc_id, 4);

        assert!(!immutable_segment
            .remove(1, num_vectors as u128)
            .await
            .expect("Failed to invalidate"));
    }
}
