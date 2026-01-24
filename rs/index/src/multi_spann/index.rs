use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use anyhow::{anyhow, Result};
use async_lock::RwLock;
use config::search_params::SearchParams;
use dashmap::DashMap;
use memmap2::Mmap;
use odht::HashTableOwned;
use quantization::quantization::Quantizer;
use utils::file_io::env::Env;

use super::user_index_info::HashConfig;
use crate::ivf::files::invalidated_ids::{InvalidatedIdsStorage, InvalidatedUserDocId};
use crate::query::planner::Planner;
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
    pending_invalidations: DashMap<u128, HashSet<u128>>,
    num_features: usize,
    env: Arc<Box<dyn Env>>,
}

impl<Q: Quantizer> MultiSpannIndex<Q> {
    /// Creates a new `MultiSpannIndex` with a required Env for asynchronous I/O.
    ///
    /// # Arguments
    /// * `base_directory` - The base directory where user indices are stored.
    /// * `user_index_info_mmap` - Mmapped memory containing user index mapping information.
    /// * `num_features` - The number of features in the vectors.
    /// * `env` - An Arc<dyn Env> for file I/O.
    ///
    /// # Returns
    /// * `Result<Self>` - A new `MultiSpannIndex` instance or an error.
    pub async fn new(
        base_directory: String,
        user_index_info_mmap: Mmap,
        num_features: usize,
        env: Arc<Box<dyn Env>>,
    ) -> Result<Self> {
        let user_index_infos = HashTableOwned::from_raw_bytes(&user_index_info_mmap).unwrap();
        let invalidated_ids_directory = format!("{base_directory}/invalidated_ids_storage");
        let invalidated_ids_storage = InvalidatedIdsStorage::read(&invalidated_ids_directory)?;
        let index = Self {
            base_directory,
            user_to_spann: DashMap::new(),
            user_index_info_mmap,
            user_index_infos,
            pending_invalidations: DashMap::new(),
            invalidated_ids_storage: RwLock::new(invalidated_ids_storage),
            num_features,
            env,
        };

        {
            let invalidated_ids = index.invalidated_ids_storage.read().await;
            for invalidated_id in invalidated_ids.iter() {
                index
                    .pending_invalidations
                    .entry(invalidated_id.user_id)
                    .or_default()
                    .insert(invalidated_id.doc_id);
                if let Some(spann_index) = index.user_to_spann.get(&invalidated_id.user_id) {
                    let _ = spann_index.invalidate(invalidated_id.doc_id).await;
                }
            }
        }

        Ok(index)
    }

    /// Returns a list of all user IDs present in the multi-user index.
    ///
    /// # Returns
    /// * `Vec<u128>` - A vector of 128-bit user IDs.
    pub fn user_ids(&self) -> Vec<u128> {
        let mut user_ids = Vec::new();
        for (key, _) in self.user_index_infos.iter() {
            user_ids.push(key);
        }
        user_ids
    }

    /// Retrieves the `Spann` index for a specific user, creating it if it doesn't exist in the cache.
    ///
    /// # Arguments
    /// * `user_id` - The ID of the user whose index to retrieve.
    ///
    /// # Returns
    /// * `Result<Arc<Spann<Q>>>` - The user's SPANN index or an error if not found.
    pub async fn get_or_create_index(&self, user_id: u128) -> Result<Arc<Spann<Q>>> {
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
        );

        let index = reader.read::<Q>(self.env.clone()).await?;

        if let Some(invalidated_docs) = self.pending_invalidations.get(&user_id) {
            let doc_ids: Vec<u128> = invalidated_docs.iter().cloned().collect();
            index.invalidate_batch(&doc_ids).await;
        }

        let arc_index = Arc::new(index);
        self.user_to_spann.insert(user_id, arc_index.clone());

        Ok(arc_index)
    }

    /// Returns an iterator over the valid documents for a specific user.
    ///
    /// # Arguments
    /// * `user_id` - The ID of the user.
    ///
    /// # Returns
    /// * `Option<SpannIter<Q>>` - An iterator if the user exists, otherwise `None`.
    pub async fn iter_for_user(&self, user_id: u128) -> Option<SpannIter<Q>> {
        match self.get_or_create_index(user_id).await {
            Ok(index) => Some(SpannIter::new(Arc::clone(&index))),
            Err(_) => None,
        }
    }

    /// Calculates the total size of all index files on disk in bytes.
    ///
    /// # Returns
    /// * `u64` - The total size in bytes.
    pub fn size_in_bytes(&self) -> u64 {
        // Compute the size of all files in the base_directory
        let mut size = 0;
        for entry in std::fs::read_dir(self.base_directory.clone()).unwrap() {
            size += std::fs::metadata(entry.unwrap().path()).unwrap().len();
        }
        size
    }

    /// Invalidates a document for a specific user.
    ///
    /// # Arguments
    /// * `user_id` - The ID of the user.
    /// * `doc_id` - The ID of the document to invalidate.
    ///
    /// # Returns
    /// * `Result<bool>` - `true` if the document was successfully invalidated, or an error.
    pub async fn invalidate(&self, user_id: u128, doc_id: u128) -> Result<bool> {
        let index = self.get_or_create_index(user_id).await?;
        let effectively_invalidated = index.invalidate(doc_id).await;
        if effectively_invalidated {
            self.pending_invalidations
                .entry(user_id)
                .or_default()
                .insert(doc_id);
            self.invalidated_ids_storage
                .write()
                .await
                .invalidate(user_id, doc_id)?;
        }
        Ok(effectively_invalidated)
    }

    /// Invalidates a batch of documents across multiple users.
    ///
    /// # Arguments
    /// * `user_to_doc_ids` - A mapping from user IDs to lists of document IDs to invalidate.
    ///
    /// # Returns
    /// * `Result<usize>` - The total number of documents effectively invalidated.
    pub async fn invalidate_batch(
        &self,
        user_to_doc_ids: &HashMap<u128, Vec<u128>>,
    ) -> Result<usize> {
        let mut effectively_invalidated_pairs = Vec::new();

        let mut total_effectively_invalidated = 0;

        for (user_id, doc_ids) in user_to_doc_ids {
            let index = self.get_or_create_index(*user_id).await?;

            let effectively_invalidated_doc_ids = index.invalidate_batch(doc_ids).await;
            total_effectively_invalidated += effectively_invalidated_doc_ids.len();

            effectively_invalidated_pairs.extend(effectively_invalidated_doc_ids.into_iter().map(
                |doc_id| InvalidatedUserDocId {
                    user_id: *user_id,
                    doc_id,
                },
            ));
        }

        if !effectively_invalidated_pairs.is_empty() {
            for pair in &effectively_invalidated_pairs {
                self.pending_invalidations
                    .entry(pair.user_id)
                    .or_default()
                    .insert(pair.doc_id);
            }
            let mut invalidated_ids_storage_write = self.invalidated_ids_storage.write().await;
            invalidated_ids_storage_write.invalidate_batch(&effectively_invalidated_pairs)?;
        }

        Ok(total_effectively_invalidated)
    }

    /// Checks if a document is invalidated for a given user.
    ///
    /// # Arguments
    /// * `user_id` - The ID of the user.
    /// * `doc_id` - The ID of the document to check.
    ///
    /// # Returns
    /// * `Result<bool>` - `true` if the document is invalidated, otherwise `false`.
    pub async fn is_invalidated(&self, user_id: u128, doc_id: u128) -> Result<bool> {
        let index = self.get_or_create_index(user_id).await?;
        Ok(index.is_invalidated(doc_id).await)
    }

    /// Returns the internal point ID for a given user and document ID.
    ///
    /// # Warning
    /// This is very expensive and should only be used for testing.
    ///
    /// # Arguments
    /// * `user_id` - The ID of the user.
    /// * `doc_id` - The ID of the document.
    ///
    /// # Returns
    /// * `Option<u32>` - The internal point ID if found, otherwise `None`.
    #[cfg(test)]
    pub async fn get_point_id(&self, user_id: u128, doc_id: u128) -> Option<u32> {
        match self.get_or_create_index(user_id).await {
            Ok(index) => index.get_point_id(doc_id).await,
            Err(_) => None,
        }
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
        match self.get_or_create_index(user_id).await {
            Ok(index) => index.get_doc_id(point_id).await,
            Err(_) => None,
        }
    }

    /// Searches for the nearest neighbors of a query vector for a specific user.
    ///
    /// # Arguments
    /// * `user_id` - The ID of the user to search for.
    /// * `query` - The query vector.
    /// * `params` - Search parameters.
    /// * `planner` - An optional search planner for additional filtering.
    ///
    /// # Returns
    /// * `Option<SearchResult>` - The search results if any, otherwise `None`.
    pub async fn search_for_user(
        &self,
        user_id: u128,
        query: Vec<f32>,
        params: &SearchParams,
        planner: Option<Arc<Planner>>,
    ) -> Option<SearchResult> {
        match self.get_or_create_index(user_id).await {
            Ok(index) => index.search(query, params, planner).await,
            Err(_) => None,
        }
    }

    /// Returns the base directory of the multi-user index.
    ///
    /// # Returns
    /// * `&String` - A reference to the base directory path.
    pub fn base_directory(&self) -> &String {
        &self.base_directory
    }

    /// Returns the total number of deleted documents across all users.
    ///
    /// # Returns
    /// * `usize` - The count of invalidated documents.
    pub async fn get_deleted_docs_count(&self) -> usize {
        return self.invalidated_ids_storage.read().await.num_entries();
    }

    /// Returns the total number of documents (including deleted ones) in the multi-user index.
    ///
    /// # Returns
    /// * `Result<usize>` - The total document count or an error if file access fails.
    pub fn get_total_docs_count(&self) -> Result<usize> {
        let num_users = self.user_index_infos.len();

        let raw_vectors_file_path = format!("{}/ivf/raw_vectors", self.base_directory);
        let raw_vectors_file = std::fs::File::open(raw_vectors_file_path)?;
        let raw_vectors_file_metadata = raw_vectors_file.metadata()?;
        let raw_vectors_file_len = raw_vectors_file_metadata.len() as usize;

        Ok(raw_vectors_file_len - num_users * 8 / (self.num_features * 4))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::fs;
    use std::sync::Arc;

    use config::collection::CollectionConfig;
    use config::search_params::SearchParams;
    use proto::muopdb::{ContainsFilter, DocumentFilter};
    use quantization::noq::NoQuantizer;
    use utils::distance::l2::L2DistanceCalculator;
    use utils::file_io::env::{DefaultEnv, Env, EnvConfig, FileType};

    use crate::ivf::files::invalidated_ids::{InvalidatedIdsStorage, InvalidatedUserDocId};
    use crate::multi_spann::builder::MultiSpannBuilder;
    use crate::multi_spann::reader::MultiSpannReader;
    use crate::multi_spann::writer::MultiSpannWriter;
    use crate::multi_terms::builder::MultiTermBuilder;
    use crate::multi_terms::index::MultiTermIndex;
    use crate::multi_terms::writer::MultiTermWriter;
    use crate::query::planner::Planner;

    fn create_env() -> Arc<Box<dyn Env>> {
        let config = EnvConfig {
            file_type: FileType::CachedStandard,
            ..EnvConfig::default()
        };
        Arc::new(Box::new(DefaultEnv::new(config)))
    }

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
                .insert(0, i, &[i as f32, i as f32, i as f32, i as f32])
                .is_ok());
        }
        assert!(multi_spann_builder
            .insert(0, num_vectors, &[1.2, 2.2, 3.2, 4.2])
            .is_ok());
        assert!(multi_spann_builder.build().is_ok());

        let multi_spann_writer = MultiSpannWriter::new(base_directory.clone());
        assert!(multi_spann_writer.write(&mut multi_spann_builder).is_ok());

        let env = create_env();
        let multi_spann_reader = MultiSpannReader::new(base_directory);
        let multi_spann_index = multi_spann_reader
            .read::<NoQuantizer<L2DistanceCalculator>>(num_features, env)
            .await
            .expect("Failed to read Multi-SPANN index");

        let query = vec![1.4, 2.4, 3.4, 4.4];
        let k = 3;
        let num_probes = 2;

        let params = SearchParams::new(k, num_probes, false);

        let results = multi_spann_index
            .search_for_user(0, query, &params, None)
            .await
            .expect("Failed to search with Multi-SPANN index");

        assert_eq!(results.id_with_scores.len(), k);
        assert_eq!(results.id_with_scores[0].doc_id, num_vectors);
        assert_eq!(results.id_with_scores[1].doc_id, 3);
        assert_eq!(results.id_with_scores[2].doc_id, 2);
    }

    #[tokio::test]
    async fn test_multi_spann_size_in_bytes() {
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
                .insert(0, i, &[i as f32, i as f32, i as f32, i as f32])
                .is_ok());
        }
        assert!(multi_spann_builder
            .insert(0, num_vectors, &[1.2, 2.2, 3.2, 4.2])
            .is_ok());
        assert!(multi_spann_builder.build().is_ok());

        let multi_spann_writer = MultiSpannWriter::new(base_directory.clone());
        assert!(multi_spann_writer.write(&mut multi_spann_builder).is_ok());

        let env = create_env();
        let multi_spann_reader = MultiSpannReader::new(base_directory);
        let multi_spann_index = multi_spann_reader
            .read::<NoQuantizer<L2DistanceCalculator>>(num_features, env)
            .await
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
                .insert(0, i, &[i as f32, i as f32, i as f32, i as f32])
                .is_ok());
        }
        assert!(multi_spann_builder
            .insert(0, num_vectors, &[1.2, 2.2, 3.2, 4.2])
            .is_ok());
        assert!(multi_spann_builder.build().is_ok());

        let multi_spann_writer = MultiSpannWriter::new(base_directory.clone());
        assert!(multi_spann_writer.write(&mut multi_spann_builder).is_ok());

        let env = create_env();
        let multi_spann_reader = MultiSpannReader::new(base_directory);
        let multi_spann_index = multi_spann_reader
            .read::<NoQuantizer<L2DistanceCalculator>>(num_features, env)
            .await
            .expect("Failed to read Multi-SPANN index");

        let query = vec![1.4, 2.4, 3.4, 4.4];
        let k = 3;
        let num_probes = 2;

        assert!(multi_spann_index
            .invalidate(0, num_vectors)
            .await
            .expect("Failed to invalidate"));
        assert!(multi_spann_index
            .is_invalidated(0, num_vectors)
            .await
            .expect("Failed to query invalidation"));

        let params = SearchParams::new(k, num_probes, false);

        let results = multi_spann_index
            .search_for_user(0, query, &params, None)
            .await
            .expect("Failed to search with Multi-SPANN index");

        assert_eq!(results.id_with_scores.len(), k);
        assert_eq!(results.id_with_scores[0].doc_id, 3);
        assert_eq!(results.id_with_scores[1].doc_id, 2);
        assert_eq!(results.id_with_scores[2].doc_id, 4);
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

        let invalidated_ids_dir = format!("{base_directory}/invalidated_ids_storage");
        assert!(fs::create_dir(&invalidated_ids_dir).is_ok());

        let mut storage = InvalidatedIdsStorage::new(&invalidated_ids_dir, 1024);

        // Invalidate a user ID and doc ID
        assert!(storage.invalidate(0, num_vectors).is_ok());

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

        let env = create_env();
        let multi_spann_reader = MultiSpannReader::new(base_directory);
        let multi_spann_index = multi_spann_reader
            .read::<NoQuantizer<L2DistanceCalculator>>(num_features, env)
            .await
            .expect("Failed to read Multi-SPANN index");
        assert!(multi_spann_index
            .is_invalidated(0, num_vectors)
            .await
            .expect("Failed to query invalidation"));
        assert_eq!(
            multi_spann_index
                .invalidated_ids_storage
                .read()
                .await
                .num_entries(),
            1
        );

        let query = vec![1.4, 2.4, 3.4, 4.4];
        let k = 3;
        let num_probes = 2;

        let params = SearchParams::new(k, num_probes, false);

        let results = multi_spann_index
            .search_for_user(0, query, &params, None)
            .await
            .expect("Failed to search with Multi-SPANN index");

        assert_eq!(results.id_with_scores.len(), k);
        assert_eq!(results.id_with_scores[0].doc_id, 3);
        assert_eq!(results.id_with_scores[1].doc_id, 2);
        assert_eq!(results.id_with_scores[2].doc_id, 4);
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
                .insert(0, i, &[i as f32, i as f32, i as f32, i as f32])
                .is_ok());
        }

        assert!(multi_spann_builder.build().is_ok());

        let multi_spann_writer = MultiSpannWriter::new(base_directory.clone());
        assert!(multi_spann_writer.write(&mut multi_spann_builder).is_ok());

        let env = create_env();
        let multi_spann_reader = MultiSpannReader::new(base_directory);
        let multi_spann_index = multi_spann_reader
            .read::<NoQuantizer<L2DistanceCalculator>>(num_features, env)
            .await
            .expect("Failed to read Multi-SPANN index");

        assert!(multi_spann_index
            .invalidate(0, 0)
            .await
            .expect("Failed to invalidate"));
        assert_eq!(
            multi_spann_index
                .invalidated_ids_storage
                .write()
                .await
                .iter()
                .collect::<Vec<_>>()
                .len(),
            1
        );

        assert!(!multi_spann_index
            .invalidate(0, num_vectors)
            .await
            .expect("Failed to invalidate"));
        assert_eq!(
            multi_spann_index
                .invalidated_ids_storage
                .write()
                .await
                .iter()
                .collect::<Vec<_>>()
                .len(),
            1
        );
    }

    #[tokio::test]
    async fn test_multi_spann_invalidate_batch() {
        let temp_dir = tempdir::TempDir::new("multi_spann_invalidate_batch_test")
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
                .insert(0, i, &[i as f32, i as f32, i as f32, i as f32])
                .is_ok());
        }

        assert!(multi_spann_builder.build().is_ok());

        let multi_spann_writer = MultiSpannWriter::new(base_directory.clone());
        assert!(multi_spann_writer.write(&mut multi_spann_builder).is_ok());

        let env = create_env();
        let multi_spann_reader = MultiSpannReader::new(base_directory);
        let multi_spann_index = multi_spann_reader
            .read::<NoQuantizer<L2DistanceCalculator>>(4, env)
            .await
            .expect("Failed to read Multi-SPANN index");

        // Batch invalidate some valid and invalid doc_ids
        let mut invalidations: HashMap<u128, Vec<u128>> = HashMap::new();
        invalidations.insert(
            0,
            vec![
                /* Valid */ 0_u128,
                1_u128,
                /* Invalid */ num_vectors,
            ],
        );

        let effectively_invalidated_count = multi_spann_index
            .invalidate_batch(&invalidations)
            .await
            .expect("Failed to batch invalidate");

        // Verify that only valid doc_ids were effectively invalidated
        assert_eq!(effectively_invalidated_count, 2); // Only (0, 0) and (0, 1) are valid

        // Verify that the invalidated IDs are stored persistently
        let invalidated_ids: Vec<_> = multi_spann_index
            .invalidated_ids_storage
            .write()
            .await
            .iter()
            .collect();

        assert_eq!(invalidated_ids.len(), 2); // Only two IDs should be stored
        assert!(invalidated_ids.contains(&InvalidatedUserDocId {
            user_id: 0,
            doc_id: 0_u128
        }));
        assert!(invalidated_ids.contains(&InvalidatedUserDocId {
            user_id: 0,
            doc_id: 1_u128
        }));

        // Attempt another batch invalidate with already invalidated IDs and new ones
        let mut new_invalidations: HashMap<u128, Vec<u128>> = HashMap::new();
        new_invalidations.insert(
            0,
            vec![
                /* Already invalidated */ 1_u128,
                /* Valid */ 2_u128,
                /* Invalid */ num_vectors + 1,
            ],
        );

        let new_effectively_invalidated_count = multi_spann_index
            .invalidate_batch(&new_invalidations)
            .await
            .expect("Failed to batch invalidate");

        // Verify that only the new valid doc_id was effectively invalidated
        assert_eq!(new_effectively_invalidated_count, 1); // Only (0, 2) is valid and not already invalidated

        // Verify that the invalidated IDs are updated persistently
        let updated_invalidated_ids: Vec<_> = multi_spann_index
            .invalidated_ids_storage
            .write()
            .await
            .iter()
            .collect();

        assert_eq!(updated_invalidated_ids.len(), 3); // Now three IDs should be stored
        assert!(updated_invalidated_ids.contains(&InvalidatedUserDocId {
            user_id: 0,
            doc_id: 0_u128
        }));
        assert!(updated_invalidated_ids.contains(&InvalidatedUserDocId {
            user_id: 0,
            doc_id: 1_u128
        }));
        assert!(updated_invalidated_ids.contains(&InvalidatedUserDocId {
            user_id: 0,
            doc_id: 2_u128
        }));
    }

    #[tokio::test]
    async fn test_multi_spann_search_with_where_document() {
        let temp_dir = tempdir::TempDir::new("multi_spann_search_with_where_document_test")
            .expect("Failed to create temporary directory");
        let base_directory = temp_dir
            .path()
            .to_str()
            .expect("Failed to convert temporary directory path to string")
            .to_string();
        let term_dir = format!("{}/terms", base_directory);
        std::fs::create_dir_all(&term_dir).unwrap();

        let num_features = 4;
        let num_vectors = 10usize;

        let mut spann_builder_config = CollectionConfig::default_test_config();
        spann_builder_config.num_features = num_features;
        // spann_builder_config.reindex = false;
        let mut multi_spann_builder =
            MultiSpannBuilder::new(spann_builder_config, base_directory.clone())
                .expect("Failed to create Multi-SPANN builder");

        let point_ids = (0..num_vectors)
            .map(|i| {
                multi_spann_builder
                    .insert(0, i as u128, &[i as f32, i as f32, i as f32, i as f32])
                    .unwrap()
            })
            .collect::<Vec<u32>>();

        let multi_builder = MultiTermBuilder::new();
        for (i, &point_id) in point_ids.iter().enumerate() {
            if i % 2 == 0 {
                multi_builder
                    .add(0, point_id, "field:even".to_string())
                    .unwrap();
            } else {
                multi_builder
                    .add(0, point_id, "field:odd".to_string())
                    .unwrap();
            }
        }

        assert!(multi_spann_builder.build().is_ok());
        assert!(multi_builder.build().is_ok());

        let multi_spann_writer = MultiSpannWriter::new(base_directory.clone());
        assert!(multi_spann_writer.write(&mut multi_spann_builder).is_ok());

        let multi_term_writer = MultiTermWriter::new_with_segment_dir(base_directory.clone());
        multi_term_writer.write(&multi_builder).unwrap();

        let multi_term_index = MultiTermIndex::new(term_dir.clone()).unwrap();

        let env = create_env();
        let multi_spann_reader = MultiSpannReader::new(base_directory);
        let multi_spann_index = multi_spann_reader
            .read::<NoQuantizer<L2DistanceCalculator>>(num_features, env)
            .await
            .expect("Failed to read Multi-SPANN index");

        let query = vec![4.4, 4.4, 4.4, 4.4];
        let k = 10;
        let num_probes = 5;

        let contains_filter = ContainsFilter {
            path: "field".to_string(),
            value: "even".to_string(),
        };
        let document_filter = DocumentFilter {
            filter: Some(proto::muopdb::document_filter::Filter::Contains(
                contains_filter,
            )),
        };

        let planner =
            Arc::new(Planner::new(0, document_filter, Arc::new(multi_term_index), None).unwrap());

        let params = SearchParams::new(k, num_probes, false);

        let results = multi_spann_index
            .search_for_user(0, query, &params, Some(planner))
            .await
            .expect("Failed to search with Multi-SPANN index");

        for result in &results.id_with_scores {
            assert_eq!(
                result.doc_id % 2,
                0,
                "Expected even doc_id, got {}",
                result.doc_id
            );
        }
    }

    #[tokio::test]
    async fn test_multi_spann_search_with_empty_where_document_filter() {
        let temp_dir = tempdir::TempDir::new("multi_spann_search_with_empty_where_document_test")
            .expect("Failed to create temporary directory");
        let base_directory = temp_dir
            .path()
            .to_str()
            .expect("Failed to convert temporary directory path to string")
            .to_string();

        let num_features = 4;
        let num_vectors = 10usize;

        let mut spann_builder_config = CollectionConfig::default_test_config();
        spann_builder_config.num_features = num_features;
        let mut multi_spann_builder =
            MultiSpannBuilder::new(spann_builder_config, base_directory.clone())
                .expect("Failed to create Multi-SPANN builder");

        let mut point_ids = Vec::new();
        for i in 0..num_vectors {
            let point_id = multi_spann_builder
                .insert(0, i as u128, &[i as f32, i as f32, i as f32, i as f32])
                .unwrap();
            point_ids.push(point_id);
        }
        assert!(multi_spann_builder.build().is_ok());

        let multi_spann_writer = MultiSpannWriter::new(base_directory.clone());
        assert!(multi_spann_writer.write(&mut multi_spann_builder).is_ok());

        let term_dir = format!("{}/terms", base_directory);
        std::fs::create_dir_all(&term_dir).unwrap();

        let multi_builder = MultiTermBuilder::new();
        for (i, &point_id) in point_ids.iter().enumerate() {
            multi_builder
                .add(0, point_id, format!("field:term{}", i))
                .unwrap();
        }
        multi_builder.build().unwrap();

        let multi_term_writer = MultiTermWriter::new_with_segment_dir(base_directory.clone());
        multi_term_writer.write(&multi_builder).unwrap();

        let multi_term_index = MultiTermIndex::new(term_dir).unwrap();

        let env = create_env();
        let multi_spann_reader = MultiSpannReader::new(base_directory);
        let multi_spann_index = multi_spann_reader
            .read::<NoQuantizer<L2DistanceCalculator>>(num_features, env)
            .await
            .expect("Failed to read Multi-SPANN index");

        let query = vec![2.4, 2.4, 2.4, 2.4];
        let k = 10;
        let num_probes = 2;

        let contains_filter = ContainsFilter {
            path: "field".to_string(),
            value: "nonexistent".to_string(),
        };
        let document_filter = DocumentFilter {
            filter: Some(proto::muopdb::document_filter::Filter::Contains(
                contains_filter,
            )),
        };

        let planner =
            Arc::new(Planner::new(0, document_filter, Arc::new(multi_term_index), None).unwrap());

        let params = SearchParams::new(k, num_probes, false);

        let results = multi_spann_index
            .search_for_user(0, query, &params, Some(planner))
            .await
            .expect("Failed to search with Multi-SPANN index");

        assert_eq!(results.id_with_scores.len(), 0);
    }
}
