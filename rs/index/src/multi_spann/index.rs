use std::sync::Arc;

use anyhow::{anyhow, Result};
use config::enums::IntSeqEncodingType;
use dashmap::DashMap;
use memmap2::Mmap;
use odht::HashTableOwned;
use parking_lot::RwLock;
use quantization::quantization::Quantizer;

use super::user_index_info::HashConfig;
use crate::ivf::files::invalidated_ids::InvalidatedIdsStorage;
use crate::spann::index::Spann;
use crate::spann::iter::SpannIter;
use crate::spann::reader::SpannReader;
use crate::utils::SearchResult;

pub struct MultiSpannIndex<Q: Quantizer> {
    base_directory: String,
    user_to_spann: DashMap<u128, Arc<Spann<Q>>>,
    #[allow(dead_code)]
    user_index_info_mmap: Mmap,
    user_index_infos: HashTableOwned<HashConfig>,
    invalidated_ids_storage: RwLock<InvalidatedIdsStorage>,
    ivf_type: IntSeqEncodingType,
}

impl<Q: Quantizer> MultiSpannIndex<Q> {
    pub fn new(
        base_directory: String,
        user_index_info_mmap: Mmap,
        ivf_type: IntSeqEncodingType,
    ) -> Result<Self> {
        let user_index_infos = HashTableOwned::from_raw_bytes(&user_index_info_mmap).unwrap();

        // Read invalidated ids
        let invalidated_ids_directory = format!("{}/invalidated_ids_storage", base_directory);

        // Initialize InvalidatedIdsStorage
        let invalidated_ids_storage = InvalidatedIdsStorage::read(&invalidated_ids_directory)?;

        // Create the MultiSpannIndex instance
        let index = Self {
            base_directory,
            user_to_spann: DashMap::new(),
            user_index_info_mmap,
            user_index_infos,
            invalidated_ids_storage: RwLock::new(invalidated_ids_storage),
            ivf_type,
        };

        // Iterate over invalidated ids and invalidate each entry. Do not call
        // MultiSpannIndex::invalidate directly as this also appends invalidated ids to the
        // storage, whereas these ids are already stored in the storage.
        {
            let invalidated_ids = index.invalidated_ids_storage.read();
            for invalidated_id in invalidated_ids.iter() {
                let spann_index = index.get_or_create_index(invalidated_id.user_id)?;
                let _ = spann_index.invalidate(invalidated_id.doc_id);
            }
        }

        Ok(index)
    }

    pub fn user_ids(&self) -> Vec<u128> {
        let mut user_ids = Vec::new();
        for (key, _) in self.user_index_infos.iter() {
            user_ids.push(key);
        }
        user_ids
    }

    fn get_or_create_index(&self, user_id: u128) -> Result<Arc<Spann<Q>>> {
        if let Some(index) = self.user_to_spann.get(&user_id) {
            return Ok(index.clone());
        }

        let index_info = self
            .user_index_infos
            .get(&user_id)
            .ok_or_else(|| anyhow!("User not found"))?;

        let reader = SpannReader::new_with_offsets(
            self.base_directory.clone(),
            index_info.centroid_index_offset as usize,
            index_info.centroid_vector_offset as usize,
            index_info.ivf_index_offset as usize,
            index_info.ivf_vectors_offset as usize,
            self.ivf_type.clone(),
        );

        let index = reader
            .read::<Q>()
            .map_err(|_| anyhow!("Failed to read index"))?;

        let arc_index = Arc::new(index);
        self.user_to_spann.insert(user_id, arc_index.clone());
        Ok(arc_index)
    }

    pub fn iter_for_user(&self, user_id: u128) -> Option<SpannIter<Q>> {
        match self.get_or_create_index(user_id) {
            Ok(index) => Some(SpannIter::new(Arc::clone(&index))),
            Err(_) => None,
        }
    }

    pub fn size_in_bytes(&self) -> u64 {
        // Compute the size of all files in the base_directory
        let mut size = 0;
        for entry in std::fs::read_dir(self.base_directory.clone()).unwrap() {
            size += std::fs::metadata(entry.unwrap().path()).unwrap().len();
        }
        size
    }

    pub fn invalidate(&self, user_id: u128, doc_id: u128) -> Result<bool> {
        let index = self.get_or_create_index(user_id)?;
        let effectively_invalidated = index.invalidate(doc_id);
        if effectively_invalidated {
            self.invalidated_ids_storage
                .write()
                .invalidate(user_id, doc_id)?;
        }
        Ok(effectively_invalidated)
    }

    pub fn is_invalidated(&self, user_id: u128, doc_id: u128) -> Result<bool> {
        let index = self.get_or_create_index(user_id)?;
        Ok(index.is_invalidated(doc_id))
    }

    pub fn get_point_id(&self, user_id: u128, doc_id: u128) -> Option<u32> {
        match self.get_or_create_index(user_id) {
            Ok(index) => index.get_point_id(doc_id),
            Err(_) => None,
        }
    }

    pub async fn search_with_id(
        &self,
        id: u128,
        query: Vec<f32>,
        k: usize,
        ef_construction: u32,
        record_pages: bool,
    ) -> Option<SearchResult> {
        match self.get_or_create_index(id) {
            Ok(index) => index.search(query, k, ef_construction, record_pages).await,
            Err(_) => None,
        }
    }

    pub fn base_directory(&self) -> &String {
        &self.base_directory
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use config::collection::CollectionConfig;
    use config::enums::IntSeqEncodingType;
    use quantization::noq::noq::NoQuantizer;
    use utils::distance::l2::L2DistanceCalculator;

    use crate::ivf::files::invalidated_ids::InvalidatedIdsStorage;
    use crate::multi_spann::builder::MultiSpannBuilder;
    use crate::multi_spann::reader::MultiSpannReader;
    use crate::multi_spann::writer::MultiSpannWriter;

    #[tokio::test]
    async fn test_multi_spann_search() {
        let temp_dir = tempdir::TempDir::new("multi_spann_search_test")
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
            .read::<NoQuantizer<L2DistanceCalculator>>(IntSeqEncodingType::PlainEncoding)
            .expect("Failed to read Multi-SPANN index");

        let query = vec![1.4, 2.4, 3.4, 4.4];
        let k = 3;
        let num_probes = 2;

        let results = multi_spann_index
            .search_with_id(0, query, k, num_probes, false)
            .await
            .expect("Failed to search with Multi-SPANN index");

        assert_eq!(results.id_with_scores.len(), k);
        assert_eq!(results.id_with_scores[0].id, num_vectors);
        assert_eq!(results.id_with_scores[1].id, 3);
        assert_eq!(results.id_with_scores[2].id, 2);
    }

    #[test]
    fn test_multi_spann_size_in_bytes() {
        let temp_dir = tempdir::TempDir::new("multi_spann_size_in_bytes_test")
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
            .read::<NoQuantizer<L2DistanceCalculator>>(IntSeqEncodingType::PlainEncoding)
            .expect("Failed to read Multi-SPANN index");

        let size_in_bytes = multi_spann_index.size_in_bytes();
        assert!(size_in_bytes >= 2000);
    }

    #[tokio::test]
    async fn test_multi_spann_search_with_invalidation() {
        let temp_dir = tempdir::TempDir::new("multi_spann_search_with_invalidation_test")
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
            .read::<NoQuantizer<L2DistanceCalculator>>(IntSeqEncodingType::PlainEncoding)
            .expect("Failed to read Multi-SPANN index");

        let query = vec![1.4, 2.4, 3.4, 4.4];
        let k = 3;
        let num_probes = 2;

        assert!(multi_spann_index
            .invalidate(0, num_vectors as u128)
            .expect("Failed to invalidate"));
        assert!(multi_spann_index
            .is_invalidated(0, num_vectors as u128)
            .expect("Failed to query invalidation"));

        let results = multi_spann_index
            .search_with_id(0, query, k, num_probes, false)
            .await
            .expect("Failed to search with Multi-SPANN index");

        assert_eq!(results.id_with_scores.len(), k);
        assert_eq!(results.id_with_scores[0].id, 3);
        assert_eq!(results.id_with_scores[1].id, 2);
        assert_eq!(results.id_with_scores[2].id, 4);
    }

    #[tokio::test]
    async fn test_multi_spann_create_with_invalidation() {
        let temp_dir = tempdir::TempDir::new("multi_spann_create_with_invalidation_test")
            .expect("Failed to create temporary directory");
        let base_directory = temp_dir
            .path()
            .to_str()
            .expect("Failed to convert temporary directory path to string")
            .to_string();

        let num_vectors = 1000;
        let num_features = 4;

        let invalidated_ids_dir = format!("{}/invalidated_ids_storage", base_directory);
        assert!(fs::create_dir(&invalidated_ids_dir).is_ok());

        let mut storage = InvalidatedIdsStorage::new(&invalidated_ids_dir, 1024);

        // Invalidate a user ID and doc ID
        assert!(storage.invalidate(0, num_vectors as u128).is_ok());

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
            .read::<NoQuantizer<L2DistanceCalculator>>(IntSeqEncodingType::PlainEncoding)
            .expect("Failed to read Multi-SPANN index");
        assert!(multi_spann_index
            .is_invalidated(0, num_vectors as u128)
            .expect("Failed to query invalidation"));
        assert_eq!(
            multi_spann_index
                .invalidated_ids_storage
                .read()
                .num_entries(),
            1
        );

        let query = vec![1.4, 2.4, 3.4, 4.4];
        let k = 3;
        let num_probes = 2;

        let results = multi_spann_index
            .search_with_id(0, query, k, num_probes, false)
            .await
            .expect("Failed to search with Multi-SPANN index");

        assert_eq!(results.id_with_scores.len(), k);
        assert_eq!(results.id_with_scores[0].id, 3);
        assert_eq!(results.id_with_scores[1].id, 2);
        assert_eq!(results.id_with_scores[2].id, 4);
    }

    #[tokio::test]
    async fn test_multi_spann_invalidate() {
        let temp_dir = tempdir::TempDir::new("multi_spann_invalidate_test")
            .expect("Failed to create temporary directory");
        let base_directory = temp_dir
            .path()
            .to_str()
            .expect("Failed to convert temporary directory path to string")
            .to_string();

        let num_vectors = 10;
        let num_features = 4;

        let mut spann_builder_config = CollectionConfig::default_test_config();
        spann_builder_config.num_features = num_features;
        let mut multi_spann_builder =
            MultiSpannBuilder::new(spann_builder_config, base_directory.clone())
                .expect("Failed to create Multi-SPANN builder");

        // Generate 10 vectors of f32, dimension 4
        for i in 0..num_vectors {
            assert!(multi_spann_builder
                .insert(0, i as u128, &vec![i as f32, i as f32, i as f32, i as f32])
                .is_ok());
        }

        assert!(multi_spann_builder.build().is_ok());

        let multi_spann_writer = MultiSpannWriter::new(base_directory.clone());
        assert!(multi_spann_writer.write(&mut multi_spann_builder).is_ok());

        let multi_spann_reader = MultiSpannReader::new(base_directory);
        let multi_spann_index = multi_spann_reader
            .read::<NoQuantizer<L2DistanceCalculator>>(IntSeqEncodingType::PlainEncoding)
            .expect("Failed to read Multi-SPANN index");

        assert!(multi_spann_index
            .invalidate(0, 0 as u128)
            .expect("Failed to invalidate"));
        assert_eq!(
            multi_spann_index
                .invalidated_ids_storage
                .write()
                .iter()
                .collect::<Vec<_>>()
                .len(),
            1
        );

        assert!(!multi_spann_index
            .invalidate(0, num_vectors as u128)
            .expect("Failed to invalidate"));
        assert_eq!(
            multi_spann_index
                .invalidated_ids_storage
                .write()
                .iter()
                .collect::<Vec<_>>()
                .len(),
            1
        );
    }
}
