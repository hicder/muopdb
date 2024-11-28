use std::collections::BinaryHeap;

use anyhow::{Context, Result};
use utils::distance::l2::L2DistanceCalculator;
use utils::DistanceCalculator;

use crate::index::Index;
use crate::posting_list::fixed_file::FixedFilePostingListStorage;
use crate::utils::{IdWithScore, SearchContext};
use crate::vector::fixed_file::FixedFileVectorStorage;

pub struct Ivf {
    // The dataset.
    pub vector_storage: FixedFileVectorStorage<f32>,
    // Each cluster is represented by a centroid vector. This is all the centroids in our IVF.
    pub centroid_storage: FixedFileVectorStorage<f32>,
    // Inverted index mapping each cluster to the vectors it contains.
    //   index: centroid index in `centroid_storage`
    //   value: list of vector indices in `vector_storage`
    pub posting_list_storage: FixedFilePostingListStorage,
    // Number of clusters.
    pub num_clusters: usize,
    // Number of probed centroids.
    pub num_probes: usize,
}

impl Ivf {
    pub fn new(
        vector_storage: FixedFileVectorStorage<f32>,
        centroid_storage: FixedFileVectorStorage<f32>,
        posting_list_storage: FixedFilePostingListStorage,
        num_clusters: usize,
        num_probes: usize,
    ) -> Self {
        Self {
            vector_storage,
            centroid_storage,
            posting_list_storage,
            num_clusters,
            num_probes,
        }
    }

    pub fn find_nearest_centroids(
        vector: &Vec<f32>,
        centroids: &FixedFileVectorStorage<f32>,
        num_probes: usize,
    ) -> Result<Vec<usize>> {
        let mut distances: Vec<(usize, f32)> = Vec::new();
        let mut context = SearchContext::new(false);
        for i in 0..centroids.num_vectors {
            let centroid = centroids
                .get(i, &mut context)
                .with_context(|| format!("Failed to get centroid at index {}", i))?;
            let dist = L2DistanceCalculator::calculate(&vector, &centroid);
            distances.push((i, dist));
        }
        distances.select_nth_unstable_by(num_probes - 1, |a, b| a.1.total_cmp(&b.1));
        Ok(distances.into_iter().map(|(idx, _)| idx).collect())
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
        if let Ok(nearest_centroids) =
            Self::find_nearest_centroids(&query.to_vec(), &self.centroid_storage, self.num_probes)
        {
            // Search in the inverted lists of the nearest centroids.
            for &centroid in &nearest_centroids {
                if let Ok(list) = self.posting_list_storage.get(centroid) {
                    for &idx in list {
                        let distance = L2DistanceCalculator::calculate(
                            query,
                            &self.vector_storage.get(idx as usize, context)?.to_vec(),
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

    use super::*;

    fn create_fixed_file_vector_storage(file_path: &String, dataset: &Vec<Vec<f32>>) -> Result<()> {
        let mut file = File::create(file_path.clone())?;

        // Write number of vectors (8 bytes)
        let num_vectors = dataset.len() as u64;
        file.write_all(&num_vectors.to_le_bytes())?;

        // Write test data
        for vector in dataset.iter() {
            for element in vector.iter() {
                file.write_all(&element.to_le_bytes())?;
            }
        }
        file.flush()?;
        Ok(())
    }

    fn create_fixed_file_posting_list_storage(
        file_path: &String,
        posting_lists: &Vec<Vec<u64>>,
    ) -> Result<()> {
        let mut file = File::create(file_path.clone())?;

        // Write number of posting_lists (8 bytes)
        let num_posting_lists = posting_lists.len();
        file.write_all(&(num_posting_lists as u64).to_le_bytes())?;

        let mut pl_offset = num_posting_lists * 2 * std::mem::size_of::<u64>();
        // Write metadata
        for posting_list in posting_lists.iter() {
            let pl_len = posting_list.len();
            file.write_all(&(pl_len as u64).to_le_bytes())?;
            file.write_all(&(pl_offset as u64).to_le_bytes())?;
            pl_offset += pl_len * std::mem::size_of::<u64>();
        }

        // Write posting lists
        for posting_list in posting_lists.iter() {
            for element in posting_list.iter() {
                file.write_all(&element.to_le_bytes())?;
            }
        }
        file.flush()?;
        Ok(())
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
        let dataset = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        assert!(create_fixed_file_vector_storage(&file_path, &dataset).is_ok());
        let storage = FixedFileVectorStorage::<f32>::new(file_path, 3)
            .expect("FixedFileVectorStorage should be created");

        let file_path = format!("{}/centroids", base_dir);
        let centroids = vec![vec![1.5, 2.5, 3.5], vec![5.5, 6.5, 7.5]];
        assert!(create_fixed_file_vector_storage(&file_path, &centroids).is_ok());
        let centroid_storage = FixedFileVectorStorage::<f32>::new(file_path, 3)
            .expect("FixedFileVectorStorage should be created");

        let file_path = format!("{}/posting_lists", base_dir);
        let posting_lists = vec![vec![0], vec![1, 2]];
        assert!(create_fixed_file_posting_list_storage(&file_path, &posting_lists).is_ok());
        let posting_list_storage = FixedFilePostingListStorage::new(file_path)
            .expect("FixedFilePostingListStorage should be created");

        let num_clusters = 2;
        let num_probes = 1;

        let ivf = Ivf::new(
            storage,
            centroid_storage,
            posting_list_storage,
            num_clusters,
            num_probes,
        );

        assert_eq!(ivf.num_clusters, num_clusters);
        assert_eq!(ivf.num_probes, num_probes);
        let cluster_0 = ivf.posting_list_storage.get(0);
        let cluster_1 = ivf.posting_list_storage.get(1);
        assert!(cluster_0.map_or(false, |list| list.contains(&0)));
        assert!(cluster_1.map_or(false, |list| list.contains(&2)));
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
        let file_path = format!("{}/centroids", base_dir);
        let vector = vec![3.0, 4.0, 5.0];
        let centroids = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        assert!(create_fixed_file_vector_storage(&file_path, &centroids).is_ok());
        let centroid_storage = FixedFileVectorStorage::<f32>::new(file_path, 3)
            .expect("FixedFileVectorStorage should be created");
        let num_probes = 2;

        let nearest = Ivf::find_nearest_centroids(&vector, &centroid_storage, num_probes)
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
        let dataset = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
            vec![2.0, 3.0, 4.0],
        ];
        assert!(create_fixed_file_vector_storage(&file_path, &dataset).is_ok());
        let storage = FixedFileVectorStorage::<f32>::new(file_path, 3)
            .expect("FixedFileVectorStorage should be created");

        let file_path = format!("{}/centroids", base_dir);
        let centroids = vec![vec![1.5, 2.5, 3.5], vec![5.5, 6.5, 7.5]];
        assert!(create_fixed_file_vector_storage(&file_path, &centroids).is_ok());
        let centroid_storage = FixedFileVectorStorage::<f32>::new(file_path, 3)
            .expect("FixedFileVectorStorage should be created");

        let file_path = format!("{}/posting_lists", base_dir);
        let posting_lists = vec![vec![0, 3], vec![1, 2]];
        assert!(create_fixed_file_posting_list_storage(&file_path, &posting_lists).is_ok());
        let posting_list_storage = FixedFilePostingListStorage::new(file_path)
            .expect("FixedFilePostingListStorage should be created");

        let num_clusters = 2;
        let num_probes = 2;

        let ivf = Ivf::new(
            storage,
            centroid_storage,
            posting_list_storage,
            num_clusters,
            num_probes,
        );

        let query = vec![2.0, 3.0, 4.0];
        let k = 2;
        let ef_construction = 10;
        let mut context = SearchContext::new(false);

        let results = ivf
            .search(&query, k, ef_construction, &mut context)
            .expect("IVF search should return a result");

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, 3); // Closest to [2.0, 3.0, 4.0]
        assert_eq!(results[1].id, 0); // Second closest to [2.0, 3.0, 4.0]
        assert!(results[0].score < results[1].score);
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
        let dataset = vec![vec![100.0, 200.0, 300.0]];
        assert!(create_fixed_file_vector_storage(&file_path, &dataset).is_ok());
        let storage = FixedFileVectorStorage::<f32>::new(file_path, 3)
            .expect("FixedFileVectorStorage should be created");

        let file_path = format!("{}/centroids", base_dir);
        let centroids = vec![vec![100.0, 200.0, 300.0]];
        assert!(create_fixed_file_vector_storage(&file_path, &centroids).is_ok());
        let centroid_storage = FixedFileVectorStorage::<f32>::new(file_path, 3)
            .expect("FixedFileVectorStorage should be created");

        let file_path = format!("{}/posting_lists", base_dir);
        let posting_lists = vec![vec![0]];
        assert!(create_fixed_file_posting_list_storage(&file_path, &posting_lists).is_ok());
        let posting_list_storage = FixedFilePostingListStorage::new(file_path)
            .expect("FixedFilePostingListStorage should be created");

        let num_clusters = 1;
        let num_probes = 1;

        let ivf = Ivf::new(
            storage,
            centroid_storage,
            posting_list_storage,
            num_clusters,
            num_probes,
        );

        let query = vec![1.0, 2.0, 3.0];
        let k = 5; // More than available results
        let ef_construction = 10;
        let mut context = SearchContext::new(false);

        let results = ivf
            .search(&query, k, ef_construction, &mut context)
            .expect("IVF search should return a result");

        assert_eq!(results.len(), 1); // Only one result available
        assert_eq!(results[0].id, 0);
    }
}
