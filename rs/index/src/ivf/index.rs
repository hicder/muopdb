use std::collections::{BinaryHeap, HashSet};
use std::marker::PhantomData;
use std::sync::RwLock;

use anyhow::{Context, Result};
use compression::compression::IntSeqDecoder;
use compression::elias_fano::ef::EliasFanoDecoder;
use compression::noc::noc::PlainDecoder;
use quantization::quantization::Quantizer;
use quantization::typing::VectorOps;
use utils::distance::l2::L2DistanceCalculator;
use utils::DistanceCalculator;

use crate::posting_list::storage::PostingListStorage;
use crate::utils::{IdWithScore, IntermediateResult, PointAndDistance, SearchResult, SearchStats};
use crate::vector::VectorStorage;

pub struct Ivf<Q: Quantizer, DC: DistanceCalculator, D: IntSeqDecoder> {
    // The dataset.
    pub vector_storage: Box<VectorStorage<Q::QuantizedT>>,

    // Each cluster is represented by a centroid vector.
    // This stores the list of centroids, along with a posting list
    // which maps each centroid to the vectors inside the same cluster
    // that it represents. The mapping is a list such that:
    //   index: centroid index to the list of centroids
    //   value: list of vector indices in `vector_storage`
    pub posting_list_storage: Box<PostingListStorage>,

    // Number of clusters.
    pub num_clusters: usize,

    pub quantizer: Q,

    pub invalid_point_ids: RwLock<HashSet<u32>>,

    _distance_calculator_marker: PhantomData<DC>,
    _decoder_marker: PhantomData<D>,
}

pub enum IvfType<Q: Quantizer> {
    L2Plain(Ivf<Q, L2DistanceCalculator, PlainDecoder>),
    L2EF(Ivf<Q, L2DistanceCalculator, EliasFanoDecoder>),
}

impl<Q: Quantizer> IvfType<Q> {
    pub async fn search_with_centroids_and_remap(
        &self,
        query: &[f32],
        nearest_centroid_ids: Vec<usize>,
        k: usize,
        record_pages: bool,
    ) -> SearchResult {
        match self {
            IvfType::L2Plain(ivf) => {
                ivf.search_with_centroids_and_remap(query, nearest_centroid_ids, k, record_pages)
                    .await
            }
            IvfType::L2EF(ivf) => {
                ivf.search_with_centroids_and_remap(query, nearest_centroid_ids, k, record_pages)
                    .await
            }
        }
    }

    /// This is very expensive and should only be used for testing.
    #[cfg(test)]
    pub fn get_point_id(&self, doc_id: u128) -> Option<u32> {
        match self {
            IvfType::L2Plain(ivf) => ivf.get_point_id(doc_id),
            IvfType::L2EF(ivf) => ivf.get_point_id(doc_id),
        }
    }

    pub fn get_vector_storage(&self) -> &VectorStorage<Q::QuantizedT> {
        match self {
            IvfType::L2Plain(ivf) => &ivf.vector_storage,
            IvfType::L2EF(ivf) => &ivf.vector_storage,
        }
    }

    pub fn get_index_storage(&self) -> &PostingListStorage {
        match self {
            IvfType::L2Plain(ivf) => &ivf.posting_list_storage,
            IvfType::L2EF(ivf) => &ivf.posting_list_storage,
        }
    }

    pub fn invalidate(&self, doc_id: u128) -> bool {
        match self {
            IvfType::L2Plain(ivf) => ivf.invalidate(doc_id),
            IvfType::L2EF(ivf) => ivf.invalidate(doc_id),
        }
    }

    pub fn invalidate_batch(&self, doc_ids: &[u128]) -> Vec<u128> {
        match self {
            IvfType::L2Plain(ivf) => ivf.invalidate_batch(doc_ids),
            IvfType::L2EF(ivf) => ivf.invalidate_batch(doc_ids),
        }
    }

    pub fn is_invalidated(&self, doc_id: u128) -> bool {
        match self {
            IvfType::L2Plain(ivf) => ivf.is_invalidated(doc_id),
            IvfType::L2EF(ivf) => ivf.is_invalidated(doc_id),
        }
    }

    pub fn num_clusters(&self) -> usize {
        match self {
            IvfType::L2Plain(ivf) => ivf.num_clusters,
            IvfType::L2EF(ivf) => ivf.num_clusters,
        }
    }

    pub fn num_vectors(&self) -> usize {
        match self {
            IvfType::L2Plain(ivf) => ivf.vector_storage.num_vectors(),
            IvfType::L2EF(ivf) => ivf.vector_storage.num_vectors(),
        }
    }
}

impl<Q: Quantizer, DC: DistanceCalculator, D: IntSeqDecoder> Ivf<Q, DC, D> {
    pub fn new(
        vector_storage: Box<VectorStorage<Q::QuantizedT>>,
        index_storage: Box<PostingListStorage>,
        num_clusters: usize,
        quantizer: Q,
    ) -> Self {
        Self {
            vector_storage,
            posting_list_storage: index_storage,
            num_clusters,
            quantizer,
            invalid_point_ids: RwLock::new(HashSet::new()),
            _distance_calculator_marker: PhantomData,
            _decoder_marker: PhantomData,
        }
    }

    pub fn find_nearest_centroids(
        vector: &[f32],
        index_storage: &PostingListStorage,
        num_probes: usize,
    ) -> Result<Vec<usize>> {
        let mut distances: Vec<(usize, f32)> = Vec::new();
        for i in 0..index_storage.header().num_clusters {
            let centroid = index_storage
                .get_centroid(i as usize)
                .with_context(|| format!("Failed to get centroid at index {}", i))?;
            let dist = DC::calculate(vector, centroid);
            distances.push((i as usize, dist));
        }
        distances.select_nth_unstable_by(num_probes - 1, |a, b| a.1.total_cmp(&b.1));
        let mut nearest_centroids: Vec<(usize, f32)> =
            distances.into_iter().take(num_probes).collect();
        nearest_centroids.sort_by(|a, b| a.1.total_cmp(&b.1));
        Ok(nearest_centroids.into_iter().map(|(idx, _)| idx).collect())
    }

    async fn scan_posting_list(
        &self,
        centroid: usize,
        query: &[f32],
        record_pages: bool,
    ) -> IntermediateResult {
        if let Ok(byte_slice) = self.posting_list_storage.get_posting_list(centroid) {
            let quantized_query = Q::QuantizedT::process_vector(query, &self.quantizer);
            let decoder =
                D::new_decoder(byte_slice).expect("Failed to create posting list decoder");

            let mut results = self
                .vector_storage
                .compute_distance_batch_async(
                    &quantized_query,
                    decoder.get_iterator(byte_slice),
                    &self.quantizer,
                    &self.invalid_point_ids,
                    record_pages,
                )
                .await
                .unwrap();
            results
                .point_and_distances
                .sort_by(|a, b| a.distance.total_cmp(&b.distance));
            results
        } else {
            IntermediateResult {
                point_and_distances: vec![],
                stats: SearchStats::new(),
            }
        }
    }

    async fn search_with_centroids(
        &self,
        query: &[f32],
        nearest_centroid_ids: Vec<usize>,
        k: usize,
        record_pages: bool,
    ) -> IntermediateResult {
        let mut heap = BinaryHeap::with_capacity(k);
        let mut final_stats = SearchStats::new();
        for &centroid in &nearest_centroid_ids {
            let results = self.scan_posting_list(centroid, query, record_pages).await;
            for id_with_score in results.point_and_distances {
                if heap.len() < k {
                    heap.push(id_with_score);
                } else if let Some(max) = heap.peek() {
                    if id_with_score < *max {
                        heap.pop();
                        heap.push(id_with_score);
                    }
                }
            }
            final_stats.merge(&results.stats);
        }

        // Convert heap to a sorted vector in ascending order.
        let mut results: Vec<PointAndDistance> = heap.into_vec();
        results.sort();

        IntermediateResult {
            point_and_distances: results,
            stats: final_stats,
        }
    }

    fn map_point_id_to_doc_id(&self, point_ids: &[PointAndDistance]) -> Vec<IdWithScore> {
        point_ids
            .iter()
            .map(|x| IdWithScore {
                doc_id: self
                    .posting_list_storage
                    .get_doc_id(x.point_id as usize)
                    .unwrap(),
                score: *x.distance,
            })
            .collect()
    }

    pub fn get_point_id(&self, doc_id: u128) -> Option<u32> {
        for point_id in 0..self.vector_storage.num_vectors() {
            if let Ok(stored_doc_id) = self.posting_list_storage.get_doc_id(point_id) {
                if stored_doc_id == doc_id {
                    return Some(point_id as u32);
                }
            }
        }
        None
    }

    pub async fn search_with_centroids_and_remap(
        &self,
        query: &[f32],
        nearest_centroid_ids: Vec<usize>,
        k: usize,
        record_pages: bool,
    ) -> SearchResult {
        let results = self
            .search_with_centroids(query, nearest_centroid_ids, k, record_pages)
            .await;
        let doc_ids = self.map_point_id_to_doc_id(&results.point_and_distances);
        SearchResult {
            id_with_scores: doc_ids,
            stats: results.stats,
        }
    }

    /// Invalidates a doc_id. Returns true if the doc_id is effectively invalidated, false
    /// otherwise (i.e. doc_id not found or had already been invalidated)
    pub fn invalidate(&self, doc_id: u128) -> bool {
        match self.get_point_id(doc_id) {
            Some(point_id) => self.invalid_point_ids.write().unwrap().insert(point_id),
            None => false,
        }
    }

    /// Invalidates a list of doc_ids. Returns a list of doc_ids that were successfully
    /// invalidated
    pub fn invalidate_batch(&self, doc_ids: &[u128]) -> Vec<u128> {
        // Collect valid point IDs and their corresponding doc_ids
        let mut valid_doc_point_pairs: Vec<(u128, u32)> = doc_ids
            .iter()
            .filter_map(|doc_id| {
                self.get_point_id(*doc_id)
                    .map(|point_id| (*doc_id, point_id))
            })
            .collect();

        // Single write lock acquisition
        let mut invalid_points_write = self.invalid_point_ids.write().unwrap();

        // Filter out already invalidated points and insert new ones
        valid_doc_point_pairs.retain(|(_, point_id)| invalid_points_write.insert(*point_id));

        // Return the list of successfully invalidated doc_ids
        valid_doc_point_pairs
            .into_iter()
            .map(|(doc_id, _)| doc_id)
            .collect()
    }

    pub fn is_invalidated(&self, doc_id: u128) -> bool {
        match self.get_point_id(doc_id) {
            Some(point_id) => self.invalid_point_ids.read().unwrap().contains(&point_id),
            None => false,
        }
    }
}

impl<Q: Quantizer, DC: DistanceCalculator, D: IntSeqDecoder> Ivf<Q, DC, D> {
    pub async fn search(
        &self,
        query: &[f32],
        k: usize,
        ef_construction: u32, // Number of probed centroids
        record_pages: bool,
    ) -> Option<SearchResult> {
        // Find the nearest centroids to the query.
        if let Ok(nearest_centroids) = Self::find_nearest_centroids(
            query,
            &self.posting_list_storage,
            ef_construction as usize,
        ) {
            // Search in the posting lists of the nearest centroids.
            let results = self
                .search_with_centroids(query, nearest_centroids, k, record_pages)
                .await;
            Some(SearchResult {
                id_with_scores: self.map_point_id_to_doc_id(&results.point_and_distances),
                stats: results.stats,
            })
        } else {
            println!("Error finding nearest centroids");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::Write;

    use anyhow::anyhow;
    use compression::noc::noc::PlainDecoder;
    use num_traits::ops::bytes::ToBytes;
    use quantization::noq::noq::NoQuantizer;
    use quantization::pq::pq::ProductQuantizer;
    use utils::distance::l2::L2DistanceCalculator;
    use utils::mem::{transmute_slice_to_u8, transmute_u8_to_slice};

    use super::*;
    use crate::posting_list::combined_file::FixedIndexFile;
    use crate::vector::fixed_file::FixedFileVectorStorage;

    fn create_fixed_file_vector_storage<T: ToBytes>(
        file_path: &str,
        dataset: &[&[T]],
    ) -> Result<()> {
        let mut file = File::create(file_path)?;

        // Write number of vectors (8 bytes)
        let num_vectors = dataset.len() as u64;
        file.write_all(&num_vectors.to_le_bytes())?;

        // Write test data
        for vector in dataset.iter() {
            for element in vector.iter() {
                file.write_all(element.to_le_bytes().as_ref())?;
            }
        }
        file.flush()?;
        Ok(())
    }

    fn create_fixed_file_index_storage(
        file_path: &str,
        doc_id_mapping: &[u128],
        centroids: &[&[f32]],
        posting_lists: &[&[u64]],
    ) -> Result<usize> {
        let mut file = File::create(file_path)?;

        let num_vectors = doc_id_mapping.len();
        let num_clusters = centroids.len();
        if num_clusters != posting_lists.len() {
            return Err(anyhow!(
                "Number of clusters mismatch: {} (centroids) vs. {} (posting lists)",
                num_clusters,
                posting_lists.len(),
            ));
        }

        // Create a test header
        let doc_id_mapping_len = size_of::<u128>() * (num_vectors + 1);
        let num_features = centroids[0].len();
        let centroids_len = size_of::<u64>() + num_features * num_clusters * size_of::<f32>();

        assert!(file.write_all(&0u8.to_le_bytes()).is_ok());
        let mut offset = 1;
        assert!(file.write_all(&(num_features as u32).to_le_bytes()).is_ok());
        offset += size_of::<u32>();
        // quantized_dimension
        assert!(file.write_all(&(num_features as u32).to_le_bytes()).is_ok());
        offset += size_of::<u32>();
        assert!(file.write_all(&(num_clusters as u32).to_le_bytes()).is_ok());
        offset += size_of::<u32>();
        assert!(file.write_all(&(num_vectors as u64).to_le_bytes()).is_ok());
        offset += size_of::<u64>();
        assert!(file
            .write_all(&(doc_id_mapping_len as u64).to_le_bytes())
            .is_ok());
        offset += size_of::<u64>();
        assert!(file
            .write_all(&(centroids_len as u64).to_le_bytes())
            .is_ok());
        offset += size_of::<u64>();
        assert!(file.write_all(&9u64.to_le_bytes()).is_ok());
        offset += size_of::<u64>();

        // Add padding to align to 16 bytes
        let mut pad: Vec<u8> = Vec::new();
        while (offset + pad.len()) % 16 != 0 {
            pad.push(0);
        }
        assert!(file.write_all(&pad).is_ok());
        offset += pad.len();

        // Write doc_id_mapping
        assert!(file.write_all(&(num_vectors as u128).to_le_bytes()).is_ok());
        offset += size_of::<u128>();
        for doc_id in doc_id_mapping.iter() {
            assert!(file.write_all(&(*doc_id).to_le_bytes()).is_ok());
            offset += size_of::<u128>();
        }

        // Write centroids
        assert!(file.write_all(&(num_clusters as u64).to_le_bytes()).is_ok());
        offset += size_of::<u64>();
        for centroid in centroids.iter() {
            assert!(file.write_all(transmute_slice_to_u8(centroid)).is_ok());
            offset += size_of::<f32>();
        }

        pad.clear();
        while (offset + pad.len()) % 8 != 0 {
            pad.push(0);
        }
        assert!(file.write_all(&pad).is_ok());
        offset += pad.len();

        // Write posting lists
        assert!(file.write_all(&(num_clusters as u64).to_le_bytes()).is_ok());
        offset += size_of::<u64>();
        // Posting list offset starts at 0 (see FileBackedAppendablePostingListStorage)
        let mut pl_offset = 0;
        for posting_list in posting_lists.iter() {
            let pl_len = size_of_val(*posting_list);
            assert!(file.write_all(&(pl_len as u64).to_le_bytes()).is_ok());
            assert!(file.write_all(&(pl_offset as u64).to_le_bytes()).is_ok());
            pl_offset += pl_len;
            offset += 2 * size_of::<u64>();
        }
        for posting_list in posting_lists.iter() {
            assert!(file.write_all(transmute_slice_to_u8(posting_list)).is_ok());
            offset += size_of_val(*posting_list);
        }

        file.flush()?;
        Ok(offset)
    }

    #[test]
    fn test_ivf_new() {
        let temp_dir =
            tempdir::TempDir::new("ivf_test").expect("Failed to create temporary directory");
        let base_dir = temp_dir
            .path()
            .to_str()
            .expect("Failed to convert temporary directory path to string")
            .to_string();
        let file_path = format!("{}/vectors", base_dir);
        let dataset: &[&[f32]] = &[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0]];
        assert!(create_fixed_file_vector_storage(&file_path, dataset).is_ok());
        let storage = FixedFileVectorStorage::<f32>::new(file_path, 3)
            .expect("FixedFileVectorStorage should be created");

        let file_path = format!("{}/index", base_dir);
        let doc_id_mapping: &[u128] = &[100, 101, 102];
        let centroids: &[&[f32]] = &[&[1.5, 2.5, 3.5], &[5.5, 6.5, 7.5]];
        let posting_lists: &[&[u64]] = &[&[0], &[1, 2]];
        assert!(create_fixed_file_index_storage(
            &file_path,
            doc_id_mapping,
            centroids,
            posting_lists
        )
        .is_ok());
        let index_storage = Box::new(PostingListStorage::FixedLocalFile(
            FixedIndexFile::new(file_path).expect("FixedIndexFile should be created"),
        ));

        let num_clusters = 2;

        let quantizer = NoQuantizer::<L2DistanceCalculator>::new(3);
        let ivf = Ivf::<_, L2DistanceCalculator, PlainDecoder>::new(
            Box::new(VectorStorage::FixedLocalFileBacked(storage)),
            index_storage,
            num_clusters,
            quantizer,
        );

        assert_eq!(ivf.num_clusters, num_clusters);
        let cluster_0 = transmute_u8_to_slice::<u64>(
            ivf.posting_list_storage
                .get_posting_list(0)
                .expect("Failed to get posting list"),
        );
        let cluster_1 = transmute_u8_to_slice::<u64>(
            ivf.posting_list_storage
                .get_posting_list(1)
                .expect("Failed to get posting list"),
        );
        assert!(cluster_0.contains(&0));
        assert!(cluster_1.contains(&2));
    }

    #[test]
    fn test_find_nearest_centroids() {
        let temp_dir = tempdir::TempDir::new("find_nearest_centroids_test")
            .expect("Failed to create temporary directory");
        let base_dir = temp_dir
            .path()
            .to_str()
            .expect("Failed to convert temporary directory path to string")
            .to_string();
        let file_path = format!("{}/index", base_dir);
        let vector = [3.0, 4.0, 5.0];
        let doc_id_mapping: &[u128] = &[100, 101, 102];
        let centroids: &[&[f32]] = &[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0]];
        let posting_lists: &[&[u64]] = &[&[0], &[1], &[2]];
        assert!(create_fixed_file_index_storage(
            &file_path,
            doc_id_mapping,
            centroids,
            posting_lists
        )
        .is_ok());
        let index_storage = Box::new(PostingListStorage::FixedLocalFile(
            FixedIndexFile::new(file_path).expect("FixedIndexFile should be created"),
        ));
        let num_probes = 2;

        let nearest =
            Ivf::<NoQuantizer<L2DistanceCalculator>, L2DistanceCalculator, PlainDecoder>::find_nearest_centroids(
                &vector,
                &index_storage,
                num_probes,
            )
            .expect("Nearest centroids should be found");

        assert_eq!(nearest[0], 1);
        assert_eq!(nearest[1], 0);
    }

    #[tokio::test]
    async fn test_ivf_search() {
        let temp_dir =
            tempdir::TempDir::new("ivf_search_test").expect("Failed to create temporary directory");
        let base_dir = temp_dir
            .path()
            .to_str()
            .expect("Failed to convert temporary directory path to string")
            .to_string();

        let file_path = format!("{}/vectors", base_dir);
        let dataset: &[&[f32]] = &[
            &[1.0, 2.0, 3.0],
            &[4.0, 5.0, 6.0],
            &[7.0, 8.0, 9.0],
            &[2.0, 3.0, 4.0],
        ];
        assert!(create_fixed_file_vector_storage(&file_path, dataset).is_ok());
        let num_features = 3;
        let storage = FixedFileVectorStorage::<f32>::new(file_path, num_features)
            .expect("FixedFileVectorStorage should be created");

        let file_path = format!("{}/index", base_dir);
        let doc_id_mapping: &[u128] = &[100, 101, 102, 103];
        let centroids: &[&[f32]] = &[&[1.5, 2.5, 3.5], &[5.5, 6.5, 7.5]];
        let posting_lists: &[&[u64]] = &[&[0u64, 3u64], &[1u64, 2u64]];
        assert!(create_fixed_file_index_storage(
            &file_path,
            doc_id_mapping,
            centroids,
            posting_lists
        )
        .is_ok());
        let index_storage = Box::new(PostingListStorage::FixedLocalFile(
            FixedIndexFile::new(file_path).expect("FixedIndexFile should be created"),
        ));

        let doc_ids_mapping = index_storage.get_doc_id(0).unwrap();
        println!("{:?}", doc_ids_mapping);

        let num_clusters = 2;
        let num_probes = 2;

        let quantizer = NoQuantizer::<L2DistanceCalculator>::new(num_features);
        let ivf: Ivf<_, L2DistanceCalculator, PlainDecoder> = Ivf::new(
            Box::new(VectorStorage::FixedLocalFileBacked(storage)),
            index_storage,
            num_clusters,
            quantizer,
        );

        let query = &[2.0, 3.0, 4.0];
        let k = 2;

        let results = ivf
            .search(query, k, num_probes, false)
            .await
            .expect("IVF search should return a result");

        assert_eq!(results.id_with_scores.len(), k);
        assert_eq!(results.id_with_scores[0].doc_id, 103); // Closest to [2.0, 3.0, 4.0]
        assert_eq!(results.id_with_scores[1].doc_id, 100); // Second closest to [2.0, 3.0, 4.0]
        assert!(results.id_with_scores[0].score < results.id_with_scores[1].score);
    }

    #[tokio::test]
    async fn test_ivf_search_with_pq() {
        let temp_dir = tempdir::TempDir::new("ivf_search_with_pq_test")
            .expect("Failed to create temporary directory");
        let base_dir = temp_dir
            .path()
            .to_str()
            .expect("Failed to convert temporary directory path to string")
            .to_string();

        let file_path = format!("{}/vectors", base_dir);
        let dataset: &[&[f32]] = &[
            &[1.0, 2.0, 3.0],
            &[4.0, 5.0, 6.0],
            &[7.0, 8.0, 9.0],
            &[2.0, 3.0, 4.0],
        ];
        let num_features = 3;
        let subvector_dimension = 1;

        let codebook = vec![1.5, 5.5, 2.5, 6.5, 3.5, 7.5];
        let quantizer = ProductQuantizer::<L2DistanceCalculator>::new(
            num_features,
            1,
            subvector_dimension,
            codebook,
            base_dir.clone(),
        )
        .expect("Can't create product quantizer");
        let quantized_dataset: Vec<Vec<u8>> = dataset
            .iter()
            .map(|x| f32::process_vector(x, &quantizer))
            .collect();

        let quantized_dataset_refs: Vec<&[u8]> =
            quantized_dataset.iter().map(|v| v.as_slice()).collect();
        let quantized_dataset_slice: &[&[u8]] = &quantized_dataset_refs;
        assert!(create_fixed_file_vector_storage(&file_path, quantized_dataset_slice).is_ok());
        let storage = FixedFileVectorStorage::<u8>::new(
            file_path,
            num_features / subvector_dimension as usize,
        )
        .expect("FixedFileVectorStorage should be created");

        let file_path = format!("{}/index", base_dir);
        let doc_id_mapping: &[u128] = &[100, 101, 102, 103];
        let centroids: &[&[f32]] = &[&[1.5, 2.5, 3.5], &[5.5, 6.5, 7.5]];
        let posting_lists: &[&[u64]] = &[&[0u64, 3u64], &[1u64, 2u64]];
        assert!(create_fixed_file_index_storage(
            &file_path,
            doc_id_mapping,
            centroids,
            posting_lists
        )
        .is_ok());
        let index_storage = Box::new(PostingListStorage::FixedLocalFile(
            FixedIndexFile::new(file_path).expect("FixedIndexFile should be created"),
        ));

        let num_clusters = 2;
        let num_probes = 2;

        let ivf: Ivf<_, L2DistanceCalculator, PlainDecoder> = Ivf::new(
            Box::new(VectorStorage::FixedLocalFileBacked(storage)),
            index_storage,
            num_clusters,
            quantizer,
        );

        let query = &[2.0, 3.0, 4.0];
        let k = 2;

        let results = ivf
            .search(query, k, num_probes, false)
            .await
            .expect("IVF search should return a result");

        assert_eq!(results.id_with_scores.len(), k);
        // This demonstrates the accuracy loss due to quantization
        assert!(results.id_with_scores[0].score == results.id_with_scores[1].score);
        assert_eq!(
            results.id_with_scores[0].doc_id + results.id_with_scores[1].doc_id,
            203
        );
        assert_eq!(
            results.id_with_scores[0]
                .doc_id
                .abs_diff(results.id_with_scores[1].doc_id),
            3
        );
    }

    #[tokio::test]
    async fn test_ivf_search_with_empty_result() {
        let temp_dir = tempdir::TempDir::new("ivf_search_error_test")
            .expect("Failed to create temporary directory");
        let base_dir = temp_dir
            .path()
            .to_str()
            .expect("Failed to convert temporary directory path to string")
            .to_string();

        let file_path = format!("{}/vectors", base_dir);
        let dataset: &[&[f32]] = &[&[100.0, 200.0, 300.0]];
        assert!(create_fixed_file_vector_storage(&file_path, dataset).is_ok());
        let num_features = 3;
        let storage = FixedFileVectorStorage::<f32>::new(file_path, num_features)
            .expect("FixedFileVectorStorage should be created");

        let file_path = format!("{}/index", base_dir);
        let doc_id_mapping: &[u128] = &[100];
        let centroids: &[&[f32]] = &[&[100.0, 200.0, 300.0]];
        let posting_lists: &[&[u64]] = &[&[0u64]];
        assert!(create_fixed_file_index_storage(
            &file_path,
            doc_id_mapping,
            centroids,
            posting_lists
        )
        .is_ok());
        let index_storage = Box::new(PostingListStorage::FixedLocalFile(
            FixedIndexFile::new(file_path).expect("FixedIndexFile should be created"),
        ));

        let num_clusters = 1;
        let num_probes = 1;

        let quantizer = NoQuantizer::<L2DistanceCalculator>::new(num_features);
        let ivf: Ivf<_, L2DistanceCalculator, PlainDecoder> = Ivf::new(
            Box::new(VectorStorage::FixedLocalFileBacked(storage)),
            index_storage,
            num_clusters,
            quantizer,
        );

        let query = &[1.0, 2.0, 3.0];
        let k = 5; // More than available results

        let results = ivf
            .search(query, k, num_probes, false)
            .await
            .expect("IVF search should return a result");

        assert_eq!(results.id_with_scores.len(), 1); // Only one result available
        assert_eq!(results.id_with_scores[0].doc_id, 100);
    }

    #[tokio::test]
    async fn test_ivf_search_invalidated_ids() {
        let temp_dir = tempdir::TempDir::new("test_ivf_search_invalidated_ids")
            .expect("Failed to create temporary directory");
        let base_dir = temp_dir
            .path()
            .to_str()
            .expect("Failed to convert temporary directory path to string")
            .to_string();

        let file_path = format!("{}/vectors", base_dir);
        let dataset: &[&[f32]] = &[
            &[1.0, 2.0, 3.0],
            &[4.0, 5.0, 6.0],
            &[7.0, 8.0, 9.0],
            &[2.0, 3.0, 4.0],
        ];
        assert!(create_fixed_file_vector_storage(&file_path, dataset).is_ok());
        let num_features = 3;
        let storage = FixedFileVectorStorage::<f32>::new(file_path, num_features)
            .expect("FixedFileVectorStorage should be created");

        let file_path = format!("{}/index", base_dir);
        let doc_id_mapping: &[u128] = &[100, 101, 102, 103];
        let centroids: &[&[f32]] = &[&[1.5, 2.5, 3.5], &[5.5, 6.5, 7.5]];
        let posting_lists: &[&[u64]] = &[&[0u64, 3u64], &[1u64, 2u64]];
        assert!(create_fixed_file_index_storage(
            &file_path,
            doc_id_mapping,
            centroids,
            posting_lists
        )
        .is_ok());
        let index_storage = Box::new(PostingListStorage::FixedLocalFile(
            FixedIndexFile::new(file_path).expect("FixedIndexFile should be created"),
        ));

        let doc_ids_mapping = index_storage.get_doc_id(0).unwrap();
        println!("{:?}", doc_ids_mapping);

        let num_clusters = 2;
        let num_probes = 2;

        let quantizer = NoQuantizer::<L2DistanceCalculator>::new(num_features);
        let ivf: Ivf<_, L2DistanceCalculator, PlainDecoder> = Ivf::new(
            Box::new(VectorStorage::FixedLocalFileBacked(storage)),
            index_storage,
            num_clusters,
            quantizer,
        );

        let query = vec![2.0, 3.0, 4.0];
        let k = 4;

        assert!(ivf.invalidate(103));
        assert!(ivf.is_invalidated(103));

        let results = ivf
            .search(&query, k, num_probes, false)
            .await
            .expect("IVF search should return a result");

        assert_eq!(results.id_with_scores.len(), k - 1);
        // doc id 103 is not in result
        assert_eq!(results.id_with_scores[0].doc_id, 100);
        assert_eq!(results.id_with_scores[1].doc_id, 101);
        assert_eq!(results.id_with_scores[2].doc_id, 102);
    }

    #[tokio::test]
    async fn test_ivf_invalidate_batch() {
        let temp_dir = tempdir::TempDir::new("test_ivf_invalidate_batch")
            .expect("Failed to create temporary directory");
        let base_dir = temp_dir
            .path()
            .to_str()
            .expect("Failed to convert temporary directory path to string")
            .to_string();

        let file_path = format!("{}/vectors", base_dir);
        let dataset: &[&[f32]] = &[
            &[1.0, 2.0, 3.0],
            &[4.0, 5.0, 6.0],
            &[7.0, 8.0, 9.0],
            &[2.0, 3.0, 4.0],
        ];
        assert!(create_fixed_file_vector_storage(&file_path, dataset).is_ok());
        let num_features = 3;
        let storage = FixedFileVectorStorage::<f32>::new(file_path, num_features)
            .expect("FixedFileVectorStorage should be created");

        let file_path = format!("{}/index", base_dir);
        let doc_id_mapping: &[u128] = &[100, 101, 102, 103];
        let centroids: &[&[f32]] = &[&[1.5, 2.5, 3.5], &[5.5, 6.5, 7.5]];
        let posting_lists: &[&[u64]] = &[&[0u64, 3u64], &[1u64, 2u64]];
        assert!(create_fixed_file_index_storage(
            &file_path,
            doc_id_mapping,
            centroids,
            posting_lists
        )
        .is_ok());
        let index_storage = Box::new(PostingListStorage::FixedLocalFile(
            FixedIndexFile::new(file_path).expect("FixedIndexFile should be created"),
        ));

        let num_clusters = 2;
        let num_probes = 2;

        let quantizer = NoQuantizer::<L2DistanceCalculator>::new(num_features);
        let ivf: Ivf<_, L2DistanceCalculator, PlainDecoder> = Ivf::new(
            Box::new(VectorStorage::FixedLocalFileBacked(storage)),
            index_storage,
            num_clusters,
            quantizer,
        );

        let query = &[2.0, 3.0, 4.0];
        let k = 4;

        // Batch invalidate doc_ids
        let invalid_doc_ids = ivf.invalidate_batch(&[101, 103]);

        // Verify that the correct doc_ids were invalidated
        assert_eq!(invalid_doc_ids.len(), 2);
        assert!(invalid_doc_ids.contains(&101));
        assert!(invalid_doc_ids.contains(&103));

        // Verify that invalidated doc_ids are excluded from the search results
        let results = ivf
            .search(query, k, num_probes, false)
            .await
            .expect("IVF search should return a result");

        assert_eq!(results.id_with_scores.len(), k - invalid_doc_ids.len());

        // Ensure invalidated doc_ids are not in the results
        for result in &results.id_with_scores {
            assert!(!invalid_doc_ids.contains(&result.doc_id));
        }

        // Ensure valid doc_ids are present in the results
        assert_eq!(results.id_with_scores[0].doc_id, 100);
        assert_eq!(results.id_with_scores[1].doc_id, 102);

        // Batch invalidate doc_ids again
        assert!(ivf.invalidate_batch(&[101, 103]).is_empty());
    }
}
