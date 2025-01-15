use std::collections::BinaryHeap;
use std::marker::PhantomData;

use anyhow::{Context, Result};
use compression::compression::IntSeqDecoder;
use quantization::quantization::Quantizer;
use quantization::typing::VectorOps;
use utils::distance::l2::L2DistanceCalculatorImpl::StreamingSIMD;
use utils::DistanceCalculator;

use crate::index::Searchable;
use crate::posting_list::combined_file::FixedIndexFile;
use crate::utils::{IdWithScore, SearchContext};
use crate::vector::fixed_file::FixedFileVectorStorage;

pub struct Ivf<Q: Quantizer, DC: DistanceCalculator, D: IntSeqDecoder<Item = u64>> {
    // The dataset.
    pub vector_storage: FixedFileVectorStorage<Q::QuantizedT>,

    // Each cluster is represented by a centroid vector.
    // This stores the list of centroids, along with a posting list
    // which maps each centroid to the vectors inside the same cluster
    // that it represents. The mapping is a list such that:
    //   index: centroid index to the list of centroids
    //   value: list of vector indices in `vector_storage`
    pub index_storage: FixedIndexFile,

    // Number of clusters.
    pub num_clusters: usize,

    pub quantizer: Q,

    _distance_calculator_marker: PhantomData<DC>,
    _decoder_marker: PhantomData<D>,
}

impl<Q: Quantizer, DC: DistanceCalculator, D: IntSeqDecoder<Item = u64>> Ivf<Q, DC, D> {
    pub fn new(
        vector_storage: FixedFileVectorStorage<Q::QuantizedT>,
        index_storage: FixedIndexFile,
        num_clusters: usize,
        quantizer: Q,
    ) -> Self {
        Self {
            vector_storage,
            index_storage,
            num_clusters,
            quantizer,
            _distance_calculator_marker: PhantomData,
            _decoder_marker: PhantomData,
        }
    }

    pub fn find_nearest_centroids(
        vector: &Vec<f32>,
        index_storage: &FixedIndexFile,
        num_probes: usize,
    ) -> Result<Vec<usize>> {
        let mut distances: Vec<(usize, f32)> = Vec::new();
        for i in 0..index_storage.header().num_clusters {
            let centroid = index_storage
                .get_centroid(i as usize)
                .with_context(|| format!("Failed to get centroid at index {}", i))?;
            let dist = DC::calculate(&vector, &centroid);
            distances.push((i as usize, dist));
        }
        distances.select_nth_unstable_by(num_probes - 1, |a, b| a.1.total_cmp(&b.1));
        let mut nearest_centroids: Vec<(usize, f32)> =
            distances.into_iter().take(num_probes).collect();
        nearest_centroids.sort_by(|a, b| a.1.total_cmp(&b.1));
        Ok(nearest_centroids.into_iter().map(|(idx, _)| idx).collect())
    }

    pub fn scan_posting_list(
        &self,
        centroid: usize,
        query: &[f32],
        context: &mut SearchContext,
    ) -> Vec<IdWithScore> {
        if let Ok(byte_slice) = self.index_storage.get_posting_list(centroid) {
            let quantized_query = Q::QuantizedT::process_vector(query, &self.quantizer);
            let mut results: Vec<IdWithScore> = Vec::new();
            let decoder =
                D::new_decoder(byte_slice).expect("Failed to create posting list decoder");
            for idx in decoder.get_iterator(byte_slice) {
                match self.vector_storage.get(idx as usize, context) {
                    Some(vector) => {
                        let distance =
                            self.quantizer
                                .distance(&quantized_query, vector, StreamingSIMD);
                        results.push(IdWithScore {
                            score: distance,
                            id: idx,
                        });
                    }
                    None => {}
                }
            }
            results
        } else {
            vec![]
        }
    }

    pub fn search_with_centroids(
        &self,
        query: &[f32],
        nearest_centroid_ids: Vec<usize>,
        k: usize,
        context: &mut SearchContext,
    ) -> Vec<IdWithScore> {
        let mut heap = BinaryHeap::with_capacity(k);
        for &centroid in &nearest_centroid_ids {
            let results = self.scan_posting_list(centroid, query, context);
            for id_with_score in results {
                if heap.len() < k {
                    heap.push(id_with_score);
                } else if let Some(max) = heap.peek() {
                    if id_with_score < *max {
                        heap.pop();
                        heap.push(id_with_score);
                    }
                }
            }
        }

        // Convert heap to a sorted vector in ascending order.
        let mut results: Vec<IdWithScore> = heap.into_vec();
        results.sort();
        results
    }

    fn map_point_id_to_doc_id(&self, point_ids: &[IdWithScore]) -> Vec<IdWithScore> {
        point_ids
            .iter()
            .map(|x| IdWithScore {
                id: self.index_storage.get_doc_id(x.id as usize).unwrap(),
                score: x.score,
            })
            .collect()
    }

    pub fn search_with_centroids_and_remap(
        &self,
        query: &[f32],
        nearest_centroid_ids: Vec<usize>,
        k: usize,
        context: &mut SearchContext,
    ) -> Vec<IdWithScore> {
        let point_ids = self.search_with_centroids(query, nearest_centroid_ids, k, context);
        let doc_ids = self.map_point_id_to_doc_id(&point_ids);
        doc_ids
    }
}

impl<Q: Quantizer, DC: DistanceCalculator, D: IntSeqDecoder<Item = u64>> Searchable
    for Ivf<Q, DC, D>
{
    fn search(
        &self,
        query: &[f32],
        k: usize,
        ef_construction: u32, // Number of probed centroids
        context: &mut SearchContext,
    ) -> Option<Vec<IdWithScore>> {
        // Find the nearest centroids to the query.
        if let Ok(nearest_centroids) = Self::find_nearest_centroids(
            &query.to_vec(),
            &self.index_storage,
            ef_construction as usize,
        ) {
            // Search in the posting lists of the nearest centroids.
            let point_ids = self.search_with_centroids(query, nearest_centroids, k, context);
            let doc_ids = self.map_point_id_to_doc_id(&point_ids);
            Some(doc_ids)
        } else {
            println!("Error finding nearest centroids");
            return None;
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

    fn create_fixed_file_vector_storage<T: ToBytes>(
        file_path: &String,
        dataset: &Vec<Vec<T>>,
    ) -> Result<()> {
        let mut file = File::create(file_path.clone())?;

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
        file_path: &String,
        doc_id_mapping: &Vec<u64>,
        centroids: &Vec<Vec<f32>>,
        posting_lists: &Vec<Vec<u64>>,
    ) -> Result<usize> {
        let mut file = File::create(file_path.clone())?;

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
        let doc_id_mapping_len = size_of::<u64>() * (num_vectors + 1);
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

        // Add padding to align to 8 bytes
        let mut pad: Vec<u8> = Vec::new();
        while (offset + pad.len()) % 8 != 0 {
            pad.push(0);
        }
        assert!(file.write_all(&pad).is_ok());
        offset += pad.len();

        // Write doc_id_mapping
        assert!(file.write_all(&(num_vectors as u64).to_le_bytes()).is_ok());
        offset += size_of::<u64>();
        for doc_id in doc_id_mapping.iter() {
            assert!(file.write_all(&(*doc_id as u64).to_le_bytes()).is_ok());
            offset += size_of::<u64>();
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
            let pl_len = posting_list.len() * size_of::<u64>();
            assert!(file.write_all(&(pl_len as u64).to_le_bytes()).is_ok());
            assert!(file.write_all(&(pl_offset as u64).to_le_bytes()).is_ok());
            pl_offset += pl_len;
            offset += 2 * size_of::<u64>();
        }
        for posting_list in posting_lists.iter() {
            assert!(file.write_all(transmute_slice_to_u8(&posting_list)).is_ok());
            offset += posting_list.len() * size_of::<u64>();
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
        let dataset: Vec<Vec<f32>> = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        assert!(create_fixed_file_vector_storage(&file_path, &dataset).is_ok());
        let storage = FixedFileVectorStorage::<f32>::new(file_path, 3)
            .expect("FixedFileVectorStorage should be created");

        let file_path = format!("{}/index", base_dir);
        let doc_id_mapping = vec![100, 101, 102];
        let centroids = vec![vec![1.5, 2.5, 3.5], vec![5.5, 6.5, 7.5]];
        let posting_lists = vec![vec![0], vec![1, 2]];
        assert!(create_fixed_file_index_storage(
            &file_path,
            &doc_id_mapping,
            &centroids,
            &posting_lists
        )
        .is_ok());
        let index_storage =
            FixedIndexFile::new(file_path).expect("FixedIndexFile should be created");

        let num_clusters = 2;

        let quantizer = NoQuantizer::<L2DistanceCalculator>::new(3);
        let ivf = Ivf::<_, L2DistanceCalculator, PlainDecoder>::new(
            storage,
            index_storage,
            num_clusters,
            quantizer,
        );

        assert_eq!(ivf.num_clusters, num_clusters);
        let cluster_0 = transmute_u8_to_slice::<u64>(
            ivf.index_storage
                .get_posting_list(0)
                .expect("Failed to get posting list"),
        );
        let cluster_1 = transmute_u8_to_slice::<u64>(
            ivf.index_storage
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
        let vector = vec![3.0, 4.0, 5.0];
        let doc_id_mapping = vec![100, 101, 102];
        let centroids = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let posting_lists = vec![vec![0], vec![1], vec![2]];
        assert!(create_fixed_file_index_storage(
            &file_path,
            &doc_id_mapping,
            &centroids,
            &posting_lists
        )
        .is_ok());
        let index_storage =
            FixedIndexFile::new(file_path).expect("FixedIndexFile should be created");
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

    #[test]
    fn test_ivf_search() {
        let temp_dir =
            tempdir::TempDir::new("ivf_search_test").expect("Failed to create temporary directory");
        let base_dir = temp_dir
            .path()
            .to_str()
            .expect("Failed to convert temporary directory path to string")
            .to_string();

        let file_path = format!("{}/vectors", base_dir);
        let dataset: Vec<Vec<f32>> = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
            vec![2.0, 3.0, 4.0],
        ];
        assert!(create_fixed_file_vector_storage(&file_path, &dataset).is_ok());
        let num_features = 3;
        let storage = FixedFileVectorStorage::<f32>::new(file_path, num_features)
            .expect("FixedFileVectorStorage should be created");

        let file_path = format!("{}/index", base_dir);
        let doc_id_mapping = vec![100, 101, 102, 103];
        let centroids = vec![vec![1.5, 2.5, 3.5], vec![5.5, 6.5, 7.5]];
        let posting_lists = vec![vec![0, 3], vec![1, 2]];
        assert!(create_fixed_file_index_storage(
            &file_path,
            &doc_id_mapping,
            &centroids,
            &posting_lists
        )
        .is_ok());
        let index_storage =
            FixedIndexFile::new(file_path).expect("FixedIndexFile should be created");

        let num_clusters = 2;
        let num_probes = 2;

        let quantizer = NoQuantizer::<L2DistanceCalculator>::new(num_features);
        let ivf: Ivf<_, L2DistanceCalculator, PlainDecoder> =
            Ivf::new(storage, index_storage, num_clusters, quantizer);

        let query = vec![2.0, 3.0, 4.0];
        let k = 2;
        let mut context = SearchContext::new(false);

        let results = ivf
            .search(&query, k, num_probes, &mut context)
            .expect("IVF search should return a result");

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, 103); // Closest to [2.0, 3.0, 4.0]
        assert_eq!(results[1].id, 100); // Second closest to [2.0, 3.0, 4.0]
        assert!(results[0].score < results[1].score);
    }

    #[test]
    fn test_ivf_search_with_pq() {
        let temp_dir = tempdir::TempDir::new("ivf_search_with_pq_test")
            .expect("Failed to create temporary directory");
        let base_dir = temp_dir
            .path()
            .to_str()
            .expect("Failed to convert temporary directory path to string")
            .to_string();

        let file_path = format!("{}/vectors", base_dir);
        let dataset = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
            vec![2.0, 3.0, 4.0],
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

        assert!(create_fixed_file_vector_storage(&file_path, &quantized_dataset).is_ok());
        let storage = FixedFileVectorStorage::<u8>::new(
            file_path,
            num_features / subvector_dimension as usize,
        )
        .expect("FixedFileVectorStorage should be created");

        let file_path = format!("{}/index", base_dir);
        let doc_id_mapping = vec![100, 101, 102, 103];
        let centroids = vec![vec![1.5, 2.5, 3.5], vec![5.5, 6.5, 7.5]];
        let posting_lists = vec![vec![0, 3], vec![1, 2]];
        assert!(create_fixed_file_index_storage(
            &file_path,
            &doc_id_mapping,
            &centroids,
            &posting_lists
        )
        .is_ok());
        let index_storage =
            FixedIndexFile::new(file_path).expect("FixedIndexFile should be created");

        let num_clusters = 2;
        let num_probes = 2;

        let ivf: Ivf<_, L2DistanceCalculator, PlainDecoder> =
            Ivf::new(storage, index_storage, num_clusters, quantizer);

        let query = vec![2.0, 3.0, 4.0];
        let k = 2;
        let mut context = SearchContext::new(false);

        let results = ivf
            .search(&query, k, num_probes, &mut context)
            .expect("IVF search should return a result");

        assert_eq!(results.len(), k);
        // This demonstrates the accuracy loss due to quantization
        assert!(results[0].score == results[1].score);
        assert_eq!(results[0].id + results[1].id, 203);
        assert_eq!(results[0].id.abs_diff(results[1].id), 3);
    }

    #[test]
    fn test_ivf_search_with_empty_result() {
        let temp_dir = tempdir::TempDir::new("ivf_search_error_test")
            .expect("Failed to create temporary directory");
        let base_dir = temp_dir
            .path()
            .to_str()
            .expect("Failed to convert temporary directory path to string")
            .to_string();

        let file_path = format!("{}/vectors", base_dir);
        let dataset: Vec<Vec<f32>> = vec![vec![100.0, 200.0, 300.0]];
        assert!(create_fixed_file_vector_storage(&file_path, &dataset).is_ok());
        let num_features = 3;
        let storage = FixedFileVectorStorage::<f32>::new(file_path, num_features)
            .expect("FixedFileVectorStorage should be created");

        let file_path = format!("{}/index", base_dir);
        let doc_id_mapping = vec![100];
        let centroids = vec![vec![100.0, 200.0, 300.0]];
        let posting_lists = vec![vec![0]];
        assert!(create_fixed_file_index_storage(
            &file_path,
            &doc_id_mapping,
            &centroids,
            &posting_lists
        )
        .is_ok());
        let index_storage =
            FixedIndexFile::new(file_path).expect("FixedIndexFile should be created");

        let num_clusters = 1;
        let num_probes = 1;

        let quantizer = NoQuantizer::<L2DistanceCalculator>::new(num_features);
        let ivf: Ivf<_, L2DistanceCalculator, PlainDecoder> =
            Ivf::new(storage, index_storage, num_clusters, quantizer);

        let query = vec![1.0, 2.0, 3.0];
        let k = 5; // More than available results
        let mut context = SearchContext::new(false);

        let results = ivf
            .search(&query, k, num_probes, &mut context)
            .expect("IVF search should return a result");

        assert_eq!(results.len(), 1); // Only one result available
        assert_eq!(results[0].id, 100);
    }
}
