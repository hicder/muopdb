use std::collections::{BinaryHeap, HashMap};

use utils::distance::l2::L2DistanceCalculator;
use utils::DistanceCalculator;

use crate::index::Index;
use crate::ivf::builder::IvfBuilder;
use crate::utils::{IdWithScore, SearchContext};
use crate::vector::fixed_file::FixedFileVectorStorage;

pub struct Ivf {
    // The dataset.
    pub vector_storage: FixedFileVectorStorage<f32>,
    // Number of clusters.
    pub num_clusters: usize,
    // Each cluster is represented by a centroid vector. This is all the centroids in our IVF.
    pub centroid_storage: FixedFileVectorStorage<f32>,
    // Inverted index mapping each cluster to the vectors it contains.
    //   key: centroid index in `centroid_storage`
    //   value: vector index in `vector_storage`
    pub inverted_lists: HashMap<usize, Vec<usize>>,
    // Number of probed centroids.
    pub num_probes: usize,
}

impl Ivf {
    pub fn new(
        vector_storage: FixedFileVectorStorage<f32>,
        centroid_storage: FixedFileVectorStorage<f32>,
        inverted_lists: HashMap<usize, Vec<usize>>,
        num_clusters: usize,
        num_probes: usize,
    ) -> Self {
        Self {
            vector_storage,
            num_clusters,
            centroid_storage,
            inverted_lists,
            num_probes,
        }
    }

    pub fn find_nearest_centroids(
        vector: &Vec<f32>,
        centroids: &FixedFileVectorStorage<f32>,
        num_probes: usize,
    ) -> Vec<usize> {
        let mut calculator = L2DistanceCalculator::new();
        let mut distances: Vec<(usize, f32)> = Vec::new();
        let mut context = SearchContext::new(false);
        for i in 0..centroids.num_vectors {
            let centroid = centroids.get(i, &mut context).unwrap();
            let dist = calculator.calculate(&vector, &centroid);
            distances.push((i, dist));
        }
        distances.select_nth_unstable_by(num_probes - 1, |a, b| a.1.partial_cmp(&b.1).unwrap());
        distances.into_iter().map(|(idx, _)| idx).collect()
    }
}

impl Index for Ivf {
    fn search(
        &self,
        query: &[f32],
        k: usize,
        _ef_construction: u32,
        context: &mut SearchContext,
    ) -> Option<Vec<IdWithScore>> {
        let mut heap = BinaryHeap::with_capacity(k);

        // Find the nearest centroids to the query.
        let nearest_centroids =
            Self::find_nearest_centroids(&query.to_vec(), &self.centroid_storage, self.num_probes);

        // Search in the inverted lists of the nearest centroids.
        for &centroid in &nearest_centroids {
            if let Some(list) = self.inverted_lists.get(&centroid) {
                for &idx in list {
                    let distance = L2DistanceCalculator::new().calculate(
                        query,
                        &self.vector_storage.get(idx, context).unwrap().to_vec(),
                    );
                    let id_with_score = IdWithScore {
                        score: distance,
                        id: idx as u64,
                    };
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
        }

        // Convert heap to a sorted vector in ascending order.
        let mut results: Vec<IdWithScore> = heap.into_vec();
        results.sort();
        Some(results)
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::Write;

    use super::*;

    fn create_fixed_file_vector_storage(file_path: &String, dataset: &Vec<Vec<f32>>) {
        let mut file = File::create(file_path.clone()).unwrap();

        // Write number of vectors (8 bytes)
        let num_vectors = dataset.len() as u64;
        file.write_all(&num_vectors.to_le_bytes()).unwrap();

        // Write test data
        for vector in dataset.iter() {
            for element in vector.iter() {
                file.write_all(&element.to_le_bytes()).unwrap();
            }
        }
        file.flush().unwrap();
    }

    #[test]
    fn test_ivf_new() {
        let tempdir = tempdir::TempDir::new("test").unwrap();
        let base_dir = tempdir.path().to_str().unwrap().to_string();
        let mut file_path = format!("{}/vectors", base_dir);
        let dataset = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        create_fixed_file_vector_storage(&file_path, &dataset);
        let storage = FixedFileVectorStorage::<f32>::new(file_path, 3).unwrap();
        file_path = format!("{}/centroids", base_dir);
        let centroids = vec![vec![1.5, 2.5, 3.5], vec![5.5, 6.5, 7.5]];
        create_fixed_file_vector_storage(&file_path, &centroids);
        let centroid_storage = FixedFileVectorStorage::<f32>::new(file_path, 3).unwrap();
        let inverted_lists = IvfBuilder::build_inverted_lists(&storage, &centroid_storage);
        let num_clusters = 2;
        let num_probes = 1;

        let ivf = Ivf::new(
            storage,
            centroid_storage,
            inverted_lists.clone(),
            num_clusters,
            num_probes,
        );

        assert_eq!(ivf.num_clusters, num_clusters);
        assert_eq!(ivf.inverted_lists, inverted_lists);
        assert_eq!(ivf.num_probes, num_probes);
        assert_eq!(ivf.inverted_lists.len(), 2);
        assert!(ivf.inverted_lists.get(&0).unwrap().contains(&0));
        assert!(ivf.inverted_lists.get(&1).unwrap().contains(&2));
    }

    #[test]
    fn test_find_nearest_centroids() {
        let vector = vec![3.0, 4.0, 5.0];
        let tempdir = tempdir::TempDir::new("test").unwrap();
        let base_dir = tempdir.path().to_str().unwrap().to_string();
        let file_path = format!("{}/centroids", base_dir);
        let centroids = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        create_fixed_file_vector_storage(&file_path, &centroids);
        let centroid_storage = FixedFileVectorStorage::<f32>::new(file_path, 3).unwrap();
        let num_probes = 2;

        let nearest = Ivf::find_nearest_centroids(&vector, &centroid_storage, num_probes);

        assert_eq!(nearest[0], 1);
        assert_eq!(nearest[1], 0);
    }

    #[test]
    fn test_ivf_search() {
        let tempdir = tempdir::TempDir::new("test").unwrap();
        let base_dir = tempdir.path().to_str().unwrap().to_string();
        let mut file_path = format!("{}/vectors", base_dir);
        let dataset = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
            vec![2.0, 3.0, 4.0],
        ];
        create_fixed_file_vector_storage(&file_path, &dataset);
        let storage = FixedFileVectorStorage::<f32>::new(file_path, 3).unwrap();
        file_path = format!("{}/centroids", base_dir);
        let centroids = vec![vec![1.5, 2.5, 3.5], vec![5.5, 6.5, 7.5]];
        create_fixed_file_vector_storage(&file_path, &centroids);
        let centroid_storage = FixedFileVectorStorage::<f32>::new(file_path, 3).unwrap();
        let inverted_lists = IvfBuilder::build_inverted_lists(&storage, &centroid_storage);
        let num_clusters = 2;
        let num_probes = 2;

        let ivf = Ivf::new(
            storage,
            centroid_storage,
            inverted_lists,
            num_clusters,
            num_probes,
        );

        let query = vec![2.0, 3.0, 4.0];
        let k = 2;
        let ef_construction = 10;
        let mut context = SearchContext::new(false);

        let results = ivf
            .search(&query, k, ef_construction, &mut context)
            .unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, 3); // Closest to [2.0, 3.0, 4.0]
        assert_eq!(results[1].id, 0); // Second closest to [2.0, 3.0, 4.0]
        assert!(results[0].score < results[1].score);
    }

    #[test]
    fn test_ivf_search_with_empty_result() {
        let tempdir = tempdir::TempDir::new("test").unwrap();
        let base_dir = tempdir.path().to_str().unwrap().to_string();
        let mut file_path = format!("{}/vectors", base_dir);
        let dataset = vec![vec![100.0, 200.0, 300.0]];
        create_fixed_file_vector_storage(&file_path, &dataset);
        let storage = FixedFileVectorStorage::<f32>::new(file_path, 3).unwrap();
        file_path = format!("{}/centroids", base_dir);
        let centroids = vec![vec![100.0, 200.0, 300.0]];
        create_fixed_file_vector_storage(&file_path, &centroids);
        let centroid_storage = FixedFileVectorStorage::<f32>::new(file_path, 3).unwrap();
        let inverted_lists = IvfBuilder::build_inverted_lists(&storage, &centroid_storage);
        let num_clusters = 1;
        let num_probes = 1;

        let ivf = Ivf::new(
            storage,
            centroid_storage,
            inverted_lists,
            num_clusters,
            num_probes,
        );

        let query = vec![1.0, 2.0, 3.0];
        let k = 5; // More than available results
        let ef_construction = 10;
        let mut context = SearchContext::new(false);

        let results = ivf
            .search(&query, k, ef_construction, &mut context)
            .unwrap();

        assert_eq!(results.len(), 1); // Only one result available
        assert_eq!(results[0].id, 0);
    }
}
