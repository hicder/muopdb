use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap};
use std::fs::{create_dir, create_dir_all};

use anyhow::{anyhow, Result};
use rand::seq::SliceRandom;
use sorted_vec::SortedVec;
use utils::distance::l2::L2DistanceCalculator;
use utils::kmeans_builder::kmeans_builder::{KMeansBuilder, KMeansVariant};
use utils::{CalculateSquared, DistanceCalculator};

use crate::posting_list::file::FileBackedAppendablePostingListStorage;
use crate::posting_list::PostingListStorage;
use crate::vector::file::FileBackedAppendableVectorStorage;
use crate::vector::VectorStorage;

pub struct IvfBuilderConfig {
    pub max_iteration: usize,
    pub batch_size: usize,
    pub num_clusters: usize,
    pub num_data_points: usize,
    pub max_clusters_per_vector: usize,
    // Threshold to add a vector to more than one cluster
    pub distance_threshold: f32,

    // Parameters for storages
    pub base_directory: String,
    pub memory_size: usize,
    pub file_size: usize,
    pub num_features: usize,

    // Parameters for clustering.
    pub tolerance: f32,
    pub max_posting_list_size: usize,
}

pub struct IvfBuilder {
    config: IvfBuilderConfig,
    vectors: Box<dyn VectorStorage<f32>>,
    centroids: Box<dyn VectorStorage<f32>>,
    posting_lists: Box<dyn for<'a> PostingListStorage<'a>>,
    doc_id_mapping: Vec<u64>,
}

// TODO(tyb): maybe merge with HNSW's one
pub struct PointAndDistance {
    pub point_id: usize,
    pub distance: f32,
}

#[derive(Debug)]
struct PostingListInfo {
    centroid: Vec<f32>,
    posting_list: Vec<usize>,
}

impl Ord for PostingListInfo {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.posting_list.len().cmp(&other.posting_list.len())
    }
}

impl PartialOrd for PostingListInfo {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for PostingListInfo {
    fn eq(&self, other: &Self) -> bool {
        self.posting_list.len() == other.posting_list.len()
    }
}

impl Eq for PostingListInfo {}

#[derive(Debug)]
struct DuplicatedVectorInstance {
    posting_list_idx: usize,
    idx: usize,
}

#[derive(Debug, Clone)]
struct StoppingPoint {
    duplicated_vector_idx: u64,
    idx_in_posting_list: usize,
}

impl PartialOrd for StoppingPoint {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.duplicated_vector_idx
            .partial_cmp(&other.duplicated_vector_idx)
    }
}

impl Ord for StoppingPoint {
    fn cmp(&self, other: &Self) -> Ordering {
        self.duplicated_vector_idx.cmp(&other.duplicated_vector_idx)
    }
}

impl PartialEq for StoppingPoint {
    fn eq(&self, other: &Self) -> bool {
        self.duplicated_vector_idx == other.duplicated_vector_idx
    }
}

impl Eq for StoppingPoint {}

#[derive(Debug)]
struct PostingListWithStoppingPoints {
    posting_list: Vec<u64>,
    stopping_points: SortedVec<StoppingPoint>,
}

impl PostingListWithStoppingPoints {
    pub fn new(posting_list: Vec<u64>, stopping_points: SortedVec<StoppingPoint>) -> Self {
        Self {
            posting_list,
            stopping_points,
        }
    }

    pub fn add_stopping_point(&mut self, duplicated_vector_idx: u64, idx_in_posting_list: usize) {
        self.stopping_points.push(StoppingPoint {
            duplicated_vector_idx,
            idx_in_posting_list,
        });
    }
}

impl PartialOrd for PostingListWithStoppingPoints {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.stopping_points.first().and_then(|sp| {
            other.stopping_points.first().map(|osp| {
                sp.duplicated_vector_idx
                    .partial_cmp(&osp.duplicated_vector_idx)
            })
        })?
    }
}

impl Ord for PostingListWithStoppingPoints {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self.stopping_points.first(), other.stopping_points.first()) {
            (Some(sp), Some(osp)) => sp.duplicated_vector_idx.cmp(&osp.duplicated_vector_idx),
            _ => panic!("Comparison is only valid when stopping_points is not empty"),
        }
    }
}

impl PartialEq for PostingListWithStoppingPoints {
    fn eq(&self, other: &Self) -> bool {
        match (self.stopping_points.first(), other.stopping_points.first()) {
            (Some(sp), Some(osp)) => sp.duplicated_vector_idx == osp.duplicated_vector_idx,
            _ => false,
        }
    }
}

impl Eq for PostingListWithStoppingPoints {}

impl IvfBuilder {
    /// Create a new IvfBuilder
    pub fn new(config: IvfBuilderConfig) -> Result<Self> {
        // Create the base directory and all parent directories if they don't exist
        create_dir_all(&config.base_directory)?;

        let vectors_path = format!("{}/builder_vector_storage", config.base_directory);
        create_dir(&vectors_path)?;

        let vectors = Box::new(FileBackedAppendableVectorStorage::<f32>::new(
            vectors_path,
            config.memory_size,
            config.file_size,
            config.num_features,
        ));

        let centroids_path = format!("{}/builder_centroid_storage", config.base_directory);
        create_dir(&centroids_path)?;

        let centroids = Box::new(FileBackedAppendableVectorStorage::<f32>::new(
            centroids_path,
            config.memory_size,
            config.file_size,
            config.num_features,
        ));

        let posting_lists_path = format!("{}/builder_posting_list_storage", config.base_directory);
        create_dir(&posting_lists_path)?;

        let posting_lists = Box::new(FileBackedAppendablePostingListStorage::new(
            posting_lists_path,
            config.memory_size,
            config.file_size,
        ));

        Ok(Self {
            config,
            vectors,
            centroids,
            posting_lists,
            doc_id_mapping: Vec::new(),
        })
    }

    pub fn config(&self) -> &IvfBuilderConfig {
        &self.config
    }

    pub fn vectors(&self) -> &dyn VectorStorage<f32> {
        &*self.vectors
    }

    pub fn doc_id_mapping(&self) -> &[u64] {
        &*self.doc_id_mapping
    }

    pub fn centroids(&self) -> &dyn VectorStorage<f32> {
        &*self.centroids
    }

    pub fn posting_lists_mut(&mut self) -> &mut dyn for<'a> PostingListStorage<'a> {
        &mut *self.posting_lists
    }

    /// Add a new vector to the dataset for training
    pub fn add_vector(&mut self, doc_id: u64, data: Vec<f32>) -> Result<()> {
        self.vectors.append(&data)?;
        self.generate_id(doc_id)?;
        Ok(())
    }

    /// Add a new centroid
    pub fn add_centroid(&mut self, centroid: &[f32]) -> Result<()> {
        self.centroids.append(centroid)?;
        Ok(())
    }

    /// Add a posting list
    pub fn add_posting_list(&mut self, posting_list: &[u64]) -> Result<()> {
        self.posting_lists.append(posting_list)?;
        Ok(())
    }

    fn generate_id(&mut self, doc_id: u64) -> Result<u32> {
        let generated_id = self.doc_id_mapping.len() as u32;
        self.doc_id_mapping.push(doc_id);
        Ok(generated_id)
    }

    fn find_nearest_centroid_inmemory(
        vector: &[f32],
        flattened_centroids: &[f32],
        dimension: usize,
    ) -> usize {
        let mut max_distance = std::f32::MIN;
        let mut centroid_index = 0;
        for i in 0..flattened_centroids.len() / dimension {
            let centroid = &flattened_centroids[i * dimension..(i + 1) * dimension];
            let dist = L2DistanceCalculator::calculate(&vector, &centroid);
            if dist > max_distance {
                max_distance = dist;
                centroid_index = i;
            }
        }
        centroid_index
    }

    fn find_nearest_centroids(
        vector: &[f32],
        centroids: &dyn VectorStorage<f32>,
        num_probes: usize,
    ) -> Result<Vec<PointAndDistance>> {
        let mut distances: Vec<PointAndDistance> = Vec::new();
        let num_centroids = centroids.len();
        for i in 0..num_centroids {
            let centroid = centroids.get(i as u32)?;
            let dist = L2DistanceCalculator::calculate_squared(&vector, &centroid);
            if dist.is_nan() {
                println!("NAN found");
            }
            distances.push(PointAndDistance {
                point_id: i,
                distance: dist,
            });
        }
        distances.select_nth_unstable_by(num_probes - 1, |a, b| a.distance.total_cmp(&b.distance));
        distances.truncate(num_probes);
        Ok(distances)
    }

    pub fn build_posting_lists(&mut self) -> Result<()> {
        let mut posting_lists: Vec<Vec<u64>> = vec![Vec::with_capacity(0); self.centroids.len()];
        // Assign vectors to nearest centroids
        for i in 0..self.vectors.len() {
            let vector = self.vectors.get(i as u32)?;
            let nearest_centroids = Self::find_nearest_centroids(
                &vector,
                self.centroids.as_ref(),
                self.config.max_clusters_per_vector,
            )?;
            // Find the nearest distance, ensuring that NaN values are treated as greater than any
            // other value
            let nearest_distance = nearest_centroids
                .iter()
                .map(|pad| pad.distance)
                .min_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Greater))
                .expect("nearest_distance should not be None");
            for point_and_distance in nearest_centroids.iter() {
                if (point_and_distance.distance - nearest_distance).abs()
                    <= nearest_distance * self.config.distance_threshold
                {
                    posting_lists[point_and_distance.point_id].push(i as u64);
                }
            }
        }

        let posting_list_storage_location = format!(
            "{}/builder_posting_list_storage",
            self.config.base_directory
        );
        create_dir(&posting_list_storage_location).unwrap_or_else(|_| {});

        self.posting_lists = Box::new(FileBackedAppendablePostingListStorage::new(
            posting_list_storage_location,
            self.config.memory_size,
            self.config.file_size,
        ));

        // Move ownership of each posting list to the posting list storage
        for posting_list in posting_lists.into_iter() {
            match self.add_posting_list(posting_list.as_ref()) {
                Ok(_) => {}
                Err(e) => {
                    println!("Error adding posting list: {e}");
                    return Err(e);
                }
            }
        }
        Ok(())
    }

    fn assign_docs_to_cluster(
        &self,
        doc_ids: Vec<usize>,
        flattened_centroids: &[f32],
    ) -> Result<Vec<PostingListInfo>> {
        let mut posting_list_infos: Vec<PostingListInfo> = Vec::new();
        posting_list_infos.reserve(doc_ids.len());
        for i in 0..flattened_centroids.len() / self.config.num_features {
            let centroid = flattened_centroids
                [i * self.config.num_features..(i + 1) * self.config.num_features]
                .to_vec();
            posting_list_infos.push(PostingListInfo {
                centroid,
                posting_list: Vec::new(),
            });
        }

        for doc_id in doc_ids {
            let vector = self.vectors.get(doc_id as u32)?;
            let nearest_centroid = Self::find_nearest_centroid_inmemory(
                &vector,
                flattened_centroids,
                self.config.num_features,
            );
            posting_list_infos[nearest_centroid]
                .posting_list
                .push(doc_id);
        }
        Ok(posting_list_infos)
    }

    fn get_flattened_dataset(&self, doc_ids: &[usize]) -> Result<Vec<f32>> {
        let mut flattened_dataset: Vec<f32> = vec![];
        for i in 0..doc_ids.len() {
            let vector = self.vectors.get(doc_ids[i] as u32)?;
            flattened_dataset.extend_from_slice(vector);
        }
        Ok(flattened_dataset)
    }

    fn cluster_docs(&self, doc_ids: Vec<usize>) -> Result<Vec<PostingListInfo>> {
        let kmeans = KMeansBuilder::new(
            self.config.num_clusters,
            self.config.max_iteration,
            self.config.tolerance,
            self.config.num_features,
            KMeansVariant::Lloyd,
        );

        let flattened_dataset = self.get_flattened_dataset(doc_ids.as_ref())?;
        let result = kmeans.fit(flattened_dataset)?;
        self.assign_docs_to_cluster(doc_ids, result.centroids.as_ref())
    }

    pub fn build_centroids(&mut self) -> Result<()> {
        // First pass to get the initial centroids
        let kmeans = KMeansBuilder::new(
            self.config.num_clusters,
            self.config.max_iteration,
            self.config.tolerance,
            self.config.num_features,
            KMeansVariant::Lloyd,
        );

        // Sample the dataset to build the first set of centroids
        let mut rng = rand::thread_rng();
        let num_input_vectors = self.vectors.len();

        // Create a vector from 0 to num_input_vectors and then shuffle it
        let mut flattened_dataset: Vec<f32> = vec![];
        let indices: Vec<usize> = (0..num_input_vectors as usize).collect();

        let selected = indices
            .choose_multiple(&mut rng, self.config.num_data_points)
            .cloned()
            .collect::<Vec<usize>>();
        selected.iter().for_each(|index| {
            flattened_dataset.extend_from_slice(self.vectors.get(*index as u32).unwrap());
        });

        let result = kmeans.fit(flattened_dataset)?;
        let posting_list_infos = self.assign_docs_to_cluster(indices, result.centroids.as_ref())?;

        // Repeatedly run kmeans on the longest posting list until no posting list is longer
        // than max_posting_list_size
        let mut heap = BinaryHeap::<PostingListInfo>::new();
        for posting_list_info in posting_list_infos {
            heap.push(posting_list_info);
        }
        while heap.len() > 0 {
            match heap.peek() {
                None => break,
                Some(longest_posting_list) => {
                    if longest_posting_list.posting_list.len() < self.config.max_posting_list_size {
                        break;
                    }
                }
            }

            let longest_posting_list = heap.pop().unwrap();
            let new_posting_list_infos =
                self.cluster_docs(longest_posting_list.posting_list.clone())?;

            // Add the new posting list infos to the heap
            for posting_list_info in new_posting_list_infos {
                heap.push(posting_list_info);
            }
        }

        // Add the centroids to the centroid storage
        // We don't need to add the posting lists to the posting list storage, since later on
        // we will add them
        for posting_list_info in heap {
            if posting_list_info.posting_list.len() == 0 {
                continue;
            }
            self.add_centroid(&posting_list_info.centroid)?;
        }

        Ok(())
    }

    pub fn build(&mut self) -> Result<()> {
        self.build_centroids()?;
        self.build_posting_lists()?;

        Ok(())
    }

    fn build_posting_lists_with_stopping_points(
        &self,
    ) -> Result<Vec<PostingListWithStoppingPoints>> {
        let mut lists_with_stopping_points = Vec::new();
        let mut occurrence_map: HashMap<u64, Vec<DuplicatedVectorInstance>> = HashMap::new();

        for list_index in 0..self.posting_lists.len() {
            let posting_list = self.posting_lists.get(list_index as u32)?;
            lists_with_stopping_points.push(PostingListWithStoppingPoints {
                posting_list: posting_list.iter().collect::<Vec<_>>(),
                stopping_points: SortedVec::new(),
            });
            for (index_in_list, vector_storage_index) in posting_list.iter().enumerate() {
                let dup_vec_instance = DuplicatedVectorInstance {
                    posting_list_idx: list_index,
                    idx: index_in_list,
                };
                occurrence_map
                    .entry(vector_storage_index)
                    .or_insert(Vec::new())
                    .push(dup_vec_instance);
            }
        }

        for (duplicated_vector_idx, posting_list_indices) in &occurrence_map {
            // Vector idx is not duplicated
            if posting_list_indices.len() == 1 {
                continue;
            }

            for dup_vec_instance in posting_list_indices {
                lists_with_stopping_points[dup_vec_instance.posting_list_idx]
                    .add_stopping_point(*duplicated_vector_idx, dup_vec_instance.idx);
            }
        }

        // Filter out the posting lists with non-empty stopping_points
        let filtered_lists: Vec<_> = lists_with_stopping_points
            .into_iter()
            .filter(|posting_list| !posting_list.stopping_points.is_empty())
            .collect();

        Ok(filtered_lists)
    }

    // [11,12,13]
    // [0,2,4,6,8,20]
    // [9,18,20]
    // [14,15,16,18]
    // [1,3,5,7,18,20]
    // [10,15,21]
    //
    // cur_idx = -1
    // 15 -> -1 + 2 + 1 = 2
    // 14 -> 0
    // 10 -> 1
    //
    // cur_idx = 2
    // 18 -> 2 + 6 + 1 = 9
    // 9 -> 3
    // 16 -> 4
    // 1 -> 5
    // 3 -> 6
    // 5 -> 7
    // 7 -> 8
    // 18 -> 9
    //
    // cur_idx = 9
    // 20 -> 9 + 5 + 1 = 15
    // 0 -> 10
    // 2 -> 11
    // 4 -> 12
    // 6 -> 13
    // 8 -> 14
    fn reassign_duplicated_vectors(&mut self, assigned_ids: &mut Vec<i32>) -> Result<()> {
        let mut min_heap: BinaryHeap<Reverse<PostingListWithStoppingPoints>> = BinaryHeap::from(
            self.build_posting_lists_with_stopping_points()?
                .into_iter()
                .map(Reverse)
                .collect::<Vec<_>>(),
        );

        let mut cur_idx = 0;
        while let Some(Reverse(first_posting_list)) = min_heap.pop() {
            let min_dup_vec_idx = first_posting_list.stopping_points[0].duplicated_vector_idx;

            // Collect all elements with the same duplicated_vector_idx
            let mut working_list = vec![first_posting_list];

            while let Some(Reverse(next_posting_list)) = min_heap.peek() {
                if next_posting_list.stopping_points[0].duplicated_vector_idx == min_dup_vec_idx {
                    working_list.push(min_heap.pop().unwrap().0); // Pop and unwrap safely
                } else {
                    break; // Exit if we reach a different duplicated_vector_idx
                }
            }

            // Process the collected elements
            for list_with_stopping_points in working_list {
                for (idx_in_posting_list, original_vector_idx) in
                    list_with_stopping_points.posting_list.iter().enumerate()
                {
                    // All vectors coming before the min stopping point have been reassigned, the
                    // (modified) posting list can now be reinserted to the binary heap if there
                    // are more stopping points left
                    if *original_vector_idx == min_dup_vec_idx {
                        if list_with_stopping_points.stopping_points.len() > 1 {
                            // Remove reassigned vectors from posting list, remove the min
                            // duplicated vector idx from stopping point list
                            min_heap.push(Reverse(PostingListWithStoppingPoints {
                                posting_list: list_with_stopping_points.posting_list
                                    [idx_in_posting_list + 1..]
                                    .to_vec(),
                                stopping_points: SortedVec::from_unsorted(
                                    list_with_stopping_points.stopping_points[1..].to_vec(),
                                ),
                            }));
                        }
                        break;
                    }
                    if assigned_ids[*original_vector_idx as usize] >= 0 {
                        return Err(anyhow!(
                            "Vectors that come before a stopping point should not be reassigned"
                        ));
                    }
                    assigned_ids[*original_vector_idx as usize] = cur_idx;
                    cur_idx += 1;
                }
            }
            assigned_ids[min_dup_vec_idx as usize] = cur_idx;
            cur_idx += 1;
        }

        Ok(())
    }

    pub fn cleanup(&mut self) -> Result<()> {
        let vectors_path = format!("{}/builder_vector_storage", self.config.base_directory);
        let centroids_path = format!("{}/builder_centroid_storage", self.config.base_directory);
        let posting_lists_path = format!(
            "{}/builder_posting_list_storage",
            self.config.base_directory
        );
        std::fs::remove_dir_all(&vectors_path)?;
        std::fs::remove_dir_all(&centroids_path)?;
        std::fs::remove_dir_all(&posting_lists_path)?;
        Ok(())
    }
}

// Test
#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use utils::test_utils::generate_random_vector;

    use super::*;

    fn count_files_with_prefix(directory: &PathBuf, file_name_prefix: &str) -> usize {
        let mut count = 0;

        for entry in directory.read_dir().expect("Cannot read directory") {
            let entry = entry.expect("Cannot read entry");
            let file_name = entry.file_name();
            let file_name_str = file_name.to_string_lossy();

            if file_name_str.starts_with(file_name_prefix) {
                count += 1;
            }
        }

        count
    }

    #[test]
    fn test_build_posting_lists() {
        env_logger::init();

        let temp_dir = tempdir::TempDir::new("build_posting_lists_test")
            .expect("Failed to create temporary directory");
        let base_directory = temp_dir
            .path()
            .to_str()
            .expect("Failed to convert temporary directory path to string")
            .to_string();
        let num_clusters = 2;
        let num_vectors = 6;
        let num_features = 1;
        let file_size = 4096;
        let balance_factor = 0.0;
        let max_posting_list_size = usize::MAX;
        let mut builder = IvfBuilder::new(IvfBuilderConfig {
            max_iteration: 1000,
            batch_size: 4,
            num_clusters,
            num_data_points: num_vectors,
            max_clusters_per_vector: 2,
            distance_threshold: 0.1,
            base_directory,
            memory_size: 1024,
            file_size,
            num_features,
            tolerance: balance_factor,
            max_posting_list_size,
        })
        .expect("Failed to create builder");
        // Generate 1000 vectors of f32, dimension 4
        for i in 0..num_vectors {
            builder
                .add_vector(i as u64, vec![(i + 1) as f32])
                .expect("Vector should be added");
        }

        let _ = builder.add_centroid(&[2.5]);
        let _ = builder.add_centroid(&[5.5]);

        let result = builder.build_posting_lists();
        assert!(result.is_ok());

        assert_eq!(
            builder
                .posting_lists
                .get(0)
                .expect("Failed to get posting list")
                .iter()
                .collect::<Vec<_>>(),
            vec![0, 1, 2, 3]
        );
        assert_eq!(
            builder
                .posting_lists
                .get(1)
                .expect("Failed to get posting list")
                .iter()
                .collect::<Vec<_>>(),
            vec![3, 4, 5]
        );
    }

    #[test]
    fn test_build_posting_lists_with_stopping_points_no_duplicates() {
        let temp_dir =
            tempdir::TempDir::new("build_posting_lists_with_stopping_points_no_duplicates_test")
                .expect("Failed to create temporary directory");
        let base_directory = temp_dir
            .path()
            .to_str()
            .expect("Failed to convert temporary directory path to string")
            .to_string();
        let num_clusters = 2;
        let num_vectors = 6;
        let num_features = 1;
        let file_size = 4096;
        let balance_factor = 0.0;
        let max_posting_list_size = usize::MAX;
        let mut builder = IvfBuilder::new(IvfBuilderConfig {
            max_iteration: 1000,
            batch_size: 4,
            num_clusters,
            num_data_points: num_vectors,
            max_clusters_per_vector: 2,
            distance_threshold: 0.1,
            base_directory,
            memory_size: 1024,
            file_size,
            num_features,
            tolerance: balance_factor,
            max_posting_list_size,
        })
        .expect("Failed to create builder");

        assert!(builder.add_posting_list(&vec![1, 2, 3]).is_ok());
        assert!(builder.add_posting_list(&vec![4, 5, 6]).is_ok());
        let result = builder.build_posting_lists_with_stopping_points().unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_build_posting_lists_with_stopping_points_with_duplicates() {
        let temp_dir =
            tempdir::TempDir::new("build_posting_lists_with_stopping_points_with_duplicates_test")
                .expect("Failed to create temporary directory");
        let base_directory = temp_dir
            .path()
            .to_str()
            .expect("Failed to convert temporary directory path to string")
            .to_string();
        let num_clusters = 3;
        let num_vectors = 6;
        let num_features = 1;
        let file_size = 4096;
        let balance_factor = 0.0;
        let max_posting_list_size = usize::MAX;
        let mut builder = IvfBuilder::new(IvfBuilderConfig {
            max_iteration: 1000,
            batch_size: 4,
            num_clusters,
            num_data_points: num_vectors,
            max_clusters_per_vector: 2,
            distance_threshold: 0.1,
            base_directory,
            memory_size: 1024,
            file_size,
            num_features,
            tolerance: balance_factor,
            max_posting_list_size,
        })
        .expect("Failed to create builder");

        assert!(builder.add_posting_list(&vec![1, 2, 3]).is_ok());
        assert!(builder.add_posting_list(&vec![2, 4, 5]).is_ok());
        assert!(builder.add_posting_list(&vec![3, 6, 7]).is_ok());

        let result = builder.build_posting_lists_with_stopping_points().unwrap();

        // Expected result
        let expected_result = vec![
            PostingListWithStoppingPoints {
                posting_list: vec![1, 2, 3],
                stopping_points: SortedVec::from_unsorted(vec![
                    StoppingPoint {
                        duplicated_vector_idx: 2,
                        idx_in_posting_list: 1,
                    },
                    StoppingPoint {
                        duplicated_vector_idx: 3,
                        idx_in_posting_list: 2,
                    },
                ]),
            },
            PostingListWithStoppingPoints {
                posting_list: vec![2, 4, 5],
                stopping_points: SortedVec::from_unsorted(vec![StoppingPoint {
                    duplicated_vector_idx: 2,
                    idx_in_posting_list: 0,
                }]),
            },
            PostingListWithStoppingPoints {
                posting_list: vec![3, 6, 7],
                stopping_points: SortedVec::from_unsorted(vec![StoppingPoint {
                    duplicated_vector_idx: 3,
                    idx_in_posting_list: 0,
                }]),
            },
        ];

        assert_eq!(result, expected_result);
    }

    #[test]
    fn test_reassign_duplicated_vectors() {
        let temp_dir = tempdir::TempDir::new("reassign_duplicated_vectors_test")
            .expect("Failed to create temporary directory");
        let base_directory = temp_dir
            .path()
            .to_str()
            .expect("Failed to convert temporary directory path to string")
            .to_string();
        let num_clusters = 4;
        let num_vectors = 6;
        let num_features = 1;
        let file_size = 4096;
        let balance_factor = 0.0;
        let max_posting_list_size = usize::MAX;
        let mut builder = IvfBuilder::new(IvfBuilderConfig {
            max_iteration: 1000,
            batch_size: 4,
            num_clusters,
            num_data_points: num_vectors,
            max_clusters_per_vector: 2,
            distance_threshold: 0.1,
            base_directory,
            memory_size: 1024,
            file_size,
            num_features,
            tolerance: balance_factor,
            max_posting_list_size,
        })
        .expect("Failed to create builder");

        assert!(builder.add_posting_list(&vec![1, 3, 5, 7, 18, 20]).is_ok());
        assert!(builder.add_posting_list(&vec![9, 18, 20]).is_ok());
        assert!(builder.add_posting_list(&vec![14, 15, 16, 18]).is_ok());
        assert!(builder.add_posting_list(&vec![0, 2, 4, 6, 8, 20]).is_ok());
        assert!(builder.add_posting_list(&vec![10, 15, 21]).is_ok());

        let mut assigned_ids = vec![-1; 22];
        assert!(builder
            .reassign_duplicated_vectors(&mut assigned_ids)
            .is_ok());

        assert_eq!(assigned_ids[14], 0);
        assert_eq!(assigned_ids[10], 1);
        assert_eq!(assigned_ids[15], 2);
        assert_eq!(assigned_ids[1], 3);
        assert_eq!(assigned_ids[3], 4);
        assert_eq!(assigned_ids[5], 5);
        assert_eq!(assigned_ids[7], 6);
        assert_eq!(assigned_ids[9], 7);
        assert_eq!(assigned_ids[16], 8);
        assert_eq!(assigned_ids[18], 9);
        assert_eq!(assigned_ids[0], 10);
        assert_eq!(assigned_ids[2], 11);
        assert_eq!(assigned_ids[4], 12);
        assert_eq!(assigned_ids[6], 13);
        assert_eq!(assigned_ids[8], 14);
        assert_eq!(assigned_ids[20], 15);
        assert_eq!(assigned_ids[21], -1);
        assert_eq!(assigned_ids[11], -1);
        assert_eq!(assigned_ids[12], -1);
        assert_eq!(assigned_ids[13], -1);
    }

    #[test]
    fn test_ivf_builder() {
        let temp_dir = tempdir::TempDir::new("ivf_builder_test")
            .expect("Failed to create temporary directory");
        let base_directory = temp_dir
            .path()
            .to_str()
            .expect("Failed to convert temporary directory path to string")
            .to_string();
        let num_clusters = 10;
        let num_vectors = 1000;
        let num_features = 4;
        let file_size = 4096;
        let balance_factor = 0.0;
        let max_posting_list_size = usize::MAX;
        let mut builder = IvfBuilder::new(IvfBuilderConfig {
            max_iteration: 1000,
            batch_size: 4,
            num_clusters,
            num_data_points: num_vectors,
            max_clusters_per_vector: 1,
            distance_threshold: 0.1,
            base_directory,
            memory_size: 1024,
            file_size,
            num_features,
            tolerance: balance_factor,
            max_posting_list_size,
        })
        .expect("Failed to create builder");
        // Generate 1000 vectors of f32, dimension 4
        for i in 0..num_vectors {
            builder
                .add_vector(i as u64, generate_random_vector(num_features))
                .expect("Vector should be added");
        }

        let result = builder.build();
        assert!(result.is_ok());

        assert_eq!(builder.vectors.len(), num_vectors);
        assert_eq!(builder.centroids.len(), num_clusters);
        assert_eq!(builder.posting_lists.len(), num_clusters);

        // Total size of vectors is bigger than file size, check that they are flushed to disk
        let vectors_path =
            PathBuf::from(&builder.config.base_directory).join("builder_vector_storage");
        assert!(vectors_path.exists());
        let count = count_files_with_prefix(&vectors_path, "vector.bin.");
        assert_eq!(
            count,
            (num_vectors * num_features * std::mem::size_of::<f32>())
                .div_ceil(builder.config.file_size)
        );

        // Total size of posting lists is bigger than file size, check that they are flushed to disk
        let posting_lists_path = PathBuf::from(format!(
            "{}/builder_posting_list_storage",
            builder.config.base_directory
        ));
        assert!(posting_lists_path.exists());

        let count = count_files_with_prefix(&posting_lists_path, "posting_list.bin.");
        assert_eq!(
            count,
            (num_vectors * std::mem::size_of::<u64>() // posting list
                + num_clusters * 2 * std::mem::size_of::<u64>()) // posting list metadata
            .div_ceil(builder.config.file_size)
        );
    }
}
