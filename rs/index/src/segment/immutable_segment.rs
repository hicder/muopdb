use anyhow::{anyhow, Result};
use quantization::quantization::Quantizer;

use super::Segment;
use crate::multi_spann::index::MultiSpannIndex;
use crate::spann::iter::SpannIter;
use crate::utils::SearchResult;

/// This is an immutable segment. This usually contains a single index.
pub struct ImmutableSegment<Q: Quantizer> {
    index: MultiSpannIndex<Q>,
    name: String,
}

impl<Q: Quantizer> ImmutableSegment<Q> {
    pub fn new(index: MultiSpannIndex<Q>, name: String) -> Self {
        Self { index, name }
    }

    pub fn user_ids(&self) -> Vec<u128> {
        self.index.user_ids()
    }

    pub fn iter_for_user(&self, user_id: u128) -> Option<SpannIter<Q>> {
        self.index.iter_for_user(user_id)
    }

    pub fn size_in_bytes(&self) -> u64 {
        self.index.size_in_bytes()
    }

    pub fn get_point_id(&self, user_id: u128, doc_id: u128) -> Option<u32> {
        self.index.get_point_id(user_id, doc_id)
    }

    pub fn is_invalidated(&self, user_id: u128, doc_id: u128) -> Result<bool> {
        self.index.is_invalidated(user_id, doc_id)
    }

    pub fn num_docs(&self) -> Result<usize> {
        self.index.get_total_docs_count()
    }

    pub fn should_auto_vacuum(&self) -> bool {
        let count_deleted_documents = self.index.get_deleted_docs_count() as f64;
        if let Ok(count_all_documents) = self.num_docs() {
            let count_all_documents = count_all_documents as f64;
            if count_deleted_documents / count_all_documents > 0.1 {
                return true;
            }
            return false;
        } else {
            return false;
        }
    }
}

/// This is the implementation of Segment for ImmutableSegment.
impl<Q: Quantizer> Segment for ImmutableSegment<Q> {
    /// ImmutableSegment does not support insertion.
    fn insert(&self, _doc_id: u128, _data: &[f32]) -> Result<()> {
        Err(anyhow!("ImmutableSegment does not support insertion"))
    }

    /// ImmutableSegment does not support actual removal, we are just invalidating documents.
    fn remove(&self, user_id: u128, doc_id: u128) -> Result<bool> {
        self.index.invalidate(user_id, doc_id)
    }

    /// ImmutableSegment does not support contains.
    fn may_contain(&self, _doc_id: u128) -> bool {
        // TODO(hicder): Implement this
        return true;
    }

    fn name(&self) -> String {
        self.name.clone()
    }
}

impl<Q: Quantizer> ImmutableSegment<Q> {
    pub async fn search_for_user(
        &self,
        user_id: u128,
        query: Vec<f32>,
        k: usize,
        ef_construction: u32,
        record_pages: bool,
    ) -> Option<SearchResult> {
        self.index
            .search_for_user(user_id, query, k, ef_construction, record_pages)
            .await
    }
}

unsafe impl<Q: Quantizer> Send for ImmutableSegment<Q> {}
unsafe impl<Q: Quantizer> Sync for ImmutableSegment<Q> {}

#[cfg(test)]
mod tests {
    use config::collection::CollectionConfig;
    use config::enums::IntSeqEncodingType;
    use quantization::noq::noq::NoQuantizer;
    use utils::distance::l2::L2DistanceCalculator;

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
                .insert(0, i as u128, &vec![i as f32, i as f32, i as f32, i as f32])
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
            .read::<NoQuantizer<L2DistanceCalculator>>(
                IntSeqEncodingType::PlainEncoding,
                num_features,
            )
            .expect("Failed to read Multi-SPANN index");

        let name_for_new_segment = format!("segment_{}", rand::random::<u64>());
        let immutable_segment =
            ImmutableSegment::new(multi_spann_index, name_for_new_segment.clone());

        let query = vec![1.4, 2.4, 3.4, 4.4];
        let k = 3;
        let num_probes = 2;

        let results = immutable_segment
            .search_for_user(0, query.clone(), k, num_probes, false)
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
                .insert(0, i as u128, &vec![i as f32, i as f32, i as f32, i as f32])
                .is_ok());
        }
        for i in 0..num_vectors {
            assert!(multi_spann_builder
                .insert(1, i as u128, &vec![i as f32, i as f32, i as f32, i as f32])
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
            .read::<NoQuantizer<L2DistanceCalculator>>(
                IntSeqEncodingType::PlainEncoding,
                num_features,
            )
            .expect("Failed to read Multi-SPANN index");

        let name_for_new_segment = format!("segment_{}", rand::random::<u64>());
        let immutable_segment =
            ImmutableSegment::new(multi_spann_index, name_for_new_segment.clone());

        let query = vec![1.4, 2.4, 3.4, 4.4];
        let k = 3;
        let num_probes = 2;

        assert!(immutable_segment
            .remove(0, num_vectors as u128)
            .expect("Failed to invalidate"));

        let results = immutable_segment
            .search_for_user(0, query.clone(), k, num_probes, false)
            .await
            .expect("Failed to search with Multi-SPANN index");

        assert_eq!(results.id_with_scores.len(), k);
        assert_eq!(results.id_with_scores[0].doc_id, 3);
        assert_eq!(results.id_with_scores[1].doc_id, 2);
        assert_eq!(results.id_with_scores[2].doc_id, 4);

        assert!(!immutable_segment
            .remove(1, num_vectors as u128)
            .expect("Failed to invalidate"));
    }
}
