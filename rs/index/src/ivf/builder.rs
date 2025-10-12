use std::cmp::{max, min, Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap};
use std::fs::{create_dir, create_dir_all};
use std::io::ErrorKind;
use std::marker::PhantomData;

use anyhow::{anyhow, Result};
use log::debug;
use rand::seq::SliceRandom;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use sorted_vec::SortedVec;
use utils::distance::l2::L2DistanceCalculator;
use utils::kmeans_builder::kmeans_builder::{KMeansBuilder, KMeansVariant};
use utils::{ceil_div, CalculateSquared, DistanceCalculator};

use crate::posting_list::file::FileBackedAppendablePostingListStorage;
use crate::utils::PointAndDistance;
use crate::vector::file::FileBackedAppendableVectorStorage;

pub struct IvfBuilderConfig {
    pub max_iteration: usize,
    pub batch_size: usize,
    pub num_clusters: usize,
    pub num_data_points_for_clustering: usize,
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

pub struct IvfBuilder<D: DistanceCalculator + CalculateSquared + Send + Sync> {
    config: IvfBuilderConfig,
    vectors: FileBackedAppendableVectorStorage<f32>,
    centroids: FileBackedAppendableVectorStorage<f32>,
    posting_lists: FileBackedAppendablePostingListStorage,
    doc_id_mapping: Vec<u128>,
    valid_point_id_mapping: HashMap<u128, u32>,
    reassigned_mappings: Option<Vec<u32>>,
    _marker: PhantomData<D>,
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
struct PostingListWithStoppingPoints {
    posting_list: Vec<u64>,
    stopping_points: SortedVec<u64>,
}

impl PostingListWithStoppingPoints {
    pub fn new(posting_list: Vec<u64>, stopping_points: SortedVec<u64>) -> Self {
        Self {
            posting_list,
            stopping_points,
        }
    }

    pub fn add_stopping_point(&mut self, stopping_point: u64) {
        self.stopping_points.push(stopping_point);
    }
}

#[allow(clippy::non_canonical_partial_ord_impl)]
impl PartialOrd for PostingListWithStoppingPoints {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if let (Some(sp), Some(osp)) = (self.stopping_points.first(), other.stopping_points.first())
        {
            match sp.partial_cmp(osp) {
                Some(Ordering::Equal) => self
                    .posting_list
                    .iter()
                    .partial_cmp(other.posting_list.iter()),
                other => other,
            }
        } else {
            None
        }
    }
}

impl Ord for PostingListWithStoppingPoints {
    fn cmp(&self, other: &Self) -> Ordering {
        if let (Some(sp), Some(osp)) = (self.stopping_points.first(), other.stopping_points.first())
        {
            match sp.cmp(osp) {
                Ordering::Equal => self.posting_list.iter().cmp(other.posting_list.iter()),
                other => other,
            }
        } else {
            panic!("Comparison is only valid when stopping_points is not empty");
        }
    }
}

impl PartialEq for PostingListWithStoppingPoints {
    fn eq(&self, other: &Self) -> bool {
        if let (Some(sp), Some(osp)) = (self.stopping_points.first(), other.stopping_points.first())
        {
            sp == osp && self.posting_list.iter().eq(other.posting_list.iter())
        } else {
            false
        }
    }
}

impl Eq for PostingListWithStoppingPoints {}

impl<D: DistanceCalculator + CalculateSquared + Send + Sync> IvfBuilder<D> {
    /// Create a new IvfBuilder
    pub fn new(config: IvfBuilderConfig) -> Result<Self> {
        // Create the base directory and all parent directories if they don't exist
        create_dir_all(&config.base_directory)?;

        let vectors_path = format!("{}/builder_vector_storage", config.base_directory);
        create_dir(&vectors_path)?;

        let vectors = FileBackedAppendableVectorStorage::<f32>::new(
            vectors_path,
            config.memory_size,
            config.file_size,
            config.num_features,
        );

        let centroids_path = format!("{}/builder_centroid_storage", config.base_directory);
        create_dir(&centroids_path)?;

        let centroids = FileBackedAppendableVectorStorage::<f32>::new(
            centroids_path,
            config.memory_size,
            config.file_size,
            config.num_features,
        );

        let posting_lists_path = format!("{}/builder_posting_list_storage", config.base_directory);
        create_dir(&posting_lists_path)?;

        let posting_lists = FileBackedAppendablePostingListStorage::new(
            posting_lists_path,
            config.memory_size,
            config.file_size,
        );

        Ok(Self {
            config,
            vectors,
            centroids,
            posting_lists,
            doc_id_mapping: Vec::new(),
            valid_point_id_mapping: HashMap::new(),
            reassigned_mappings: None,
            _marker: PhantomData,
        })
    }

    pub fn config(&self) -> &IvfBuilderConfig {
        &self.config
    }

    pub fn vectors(&self) -> &FileBackedAppendableVectorStorage<f32> {
        &self.vectors
    }

    pub fn doc_id_mapping(&self) -> &[u128] {
        &self.doc_id_mapping
    }

    pub fn centroids(&self) -> &FileBackedAppendableVectorStorage<f32> {
        &self.centroids
    }

    pub fn posting_lists(&self) -> &FileBackedAppendablePostingListStorage {
        &self.posting_lists
    }

    pub fn posting_lists_mut(&mut self) -> &mut FileBackedAppendablePostingListStorage {
        &mut self.posting_lists
    }

    /// Add a new vector to the dataset for training
    pub fn add_vector(&mut self, doc_id: u128, data: &[f32]) -> Result<u32> {
        self.vectors.append(data)?;
        self.generate_id(doc_id)
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

    /// Remove the doc_id from valid_point_id_mapping, return true if doc_id is
    /// in the map and the removal takes place effectively, false otherwise
    pub fn invalidate(&mut self, doc_id: u128) -> bool {
        self.valid_point_id_mapping.remove(&doc_id).is_some()
    }

    fn generate_id(&mut self, doc_id: u128) -> Result<u32> {
        if self.doc_id_mapping.len() > u32::MAX as usize {
            return Err(anyhow!(
                "Trying to add more than 4B documents to the index, which is not allowed"
            ));
        }
        let generated_point_id = self.doc_id_mapping.len() as u32;
        self.doc_id_mapping.push(doc_id);
        self.valid_point_id_mapping
            .insert(doc_id, generated_point_id);
        Ok(generated_point_id)
    }

    fn find_nearest_centroid_inmemory(
        vector: &[f32],
        flattened_centroids: &[f32],
        dimension: usize,
    ) -> usize {
        let mut max_distance = f32::MIN;
        let mut centroid_index = 0;
        for i in 0..flattened_centroids.len() / dimension {
            let centroid = &flattened_centroids[i * dimension..(i + 1) * dimension];
            let dist = D::calculate(vector, centroid);
            if dist > max_distance {
                max_distance = dist;
                centroid_index = i;
            }
        }
        centroid_index
    }

    fn find_nearest_centroids(
        vector: &[f32],
        centroids: &FileBackedAppendableVectorStorage<f32>,
        num_probes: usize,
    ) -> Result<Vec<PointAndDistance>> {
        let mut distances: Vec<PointAndDistance> = Vec::new();
        for i in 0..centroids.num_vectors() {
            let centroid = centroids.get_no_context(i as u32)?;
            let dist = L2DistanceCalculator::calculate_squared(vector, centroid);
            distances.push(PointAndDistance::new(dist, i as u32));
        }
        distances.select_nth_unstable_by(num_probes - 1, |a, b| a.distance.total_cmp(&b.distance));
        distances.truncate(num_probes);
        Ok(distances)
    }

    pub fn get_num_valid_vectors(&self) -> usize {
        self.valid_point_id_mapping.len()
    }

    pub fn is_valid_doc_id(&self, doc_id: u128) -> bool {
        self.valid_point_id_mapping.contains_key(&doc_id)
    }

    pub fn build_posting_lists(&mut self) -> Result<()> {
        debug!("Building posting lists");

        let mut posting_lists: Vec<Vec<u64>> =
            vec![Vec::with_capacity(0); self.centroids.num_vectors()];

        // Assign vectors to nearest centroids
        let mut point_ids: Vec<u32> = self.valid_point_id_mapping.values().cloned().collect();
        point_ids.sort();

        let max_clusters_per_vector = self.config.max_clusters_per_vector;
        let posting_list_per_point: HashMap<u32, Vec<u64>> = point_ids
            .par_iter()
            .map(|point_id| {
                let nearest_centroids = Self::find_nearest_centroids(
                    self.vectors.get_no_context(*point_id).unwrap(),
                    &self.centroids,
                    max_clusters_per_vector,
                )
                .expect("Nearest centroids should not be None");
                // Find the nearest distance, ensuring that NaN values are treated as greater than any
                // other value
                let nearest_distance = nearest_centroids
                    .iter()
                    .map(|pad| pad.distance.into_inner())
                    .min_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Greater))
                    .expect("nearest_distance should not be None");
                let mut accepted_centroid_ids = vec![];
                for centroid_and_distance in nearest_centroids.iter() {
                    if (centroid_and_distance.distance - nearest_distance).abs()
                        <= nearest_distance * self.config.distance_threshold
                    {
                        accepted_centroid_ids.push(centroid_and_distance.point_id as u64);
                    }
                }
                (*point_id, accepted_centroid_ids)
            })
            .collect();

        posting_list_per_point
            .iter()
            .for_each(|(point_id, accepted_centroid_ids)| {
                accepted_centroid_ids.iter().for_each(|centroid_id| {
                    posting_lists[*centroid_id as usize].push(*point_id as u64);
                });
            });

        for posting_list in posting_lists.iter_mut() {
            posting_list.sort();
        }

        let posting_list_storage_location = format!(
            "{}/builder_posting_list_storage",
            self.config.base_directory
        );
        create_dir(&posting_list_storage_location).unwrap_or(());

        self.posting_lists = FileBackedAppendablePostingListStorage::new(
            posting_list_storage_location,
            self.config.memory_size,
            self.config.file_size,
        );

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
        point_ids: Vec<usize>,
        flattened_centroids: &[f32],
    ) -> Result<Vec<PostingListInfo>> {
        let mut posting_list_infos: Vec<PostingListInfo> = Vec::with_capacity(point_ids.len());
        for i in 0..flattened_centroids.len() / self.config.num_features {
            let centroid = flattened_centroids
                [i * self.config.num_features..(i + 1) * self.config.num_features]
                .to_vec();
            posting_list_infos.push(PostingListInfo {
                centroid,
                posting_list: Vec::new(),
            });
        }

        let num_features = self.config.num_features;
        let nearest_centroids = point_ids
            .par_iter()
            .map(|point_id| {
                Self::find_nearest_centroid_inmemory(
                    self.vectors.get_no_context(*point_id as u32).unwrap(),
                    flattened_centroids,
                    num_features,
                )
            })
            .collect::<Vec<usize>>();

        for (point_id, nearest_centroid) in point_ids.iter().zip(nearest_centroids.iter()) {
            posting_list_infos[*nearest_centroid]
                .posting_list
                .push(*point_id);
        }
        Ok(posting_list_infos)
    }

    fn get_sample_dataset_from_point_ids(
        &self,
        point_ids: &[usize],
        sample_size: usize,
    ) -> Result<Vec<f32>> {
        let mut rng = rand::thread_rng();
        let mut flattened_dataset: Vec<f32> = vec![];
        point_ids
            .choose_multiple(&mut rng, sample_size)
            .for_each(|point_id| {
                flattened_dataset
                    .extend_from_slice(self.vectors.get_no_context(*point_id as u32).unwrap());
            });
        Ok(flattened_dataset)
    }

    fn cluster_docs(
        &self,
        point_ids: Vec<usize>,
        max_posting_list_size: usize,
    ) -> Result<Vec<PostingListInfo>> {
        let num_clusters = ceil_div(point_ids.len(), max_posting_list_size);

        let num_points_for_clustering = max(
            num_clusters * 10,
            self.config.num_data_points_for_clustering,
        );
        let kmeans = KMeansBuilder::<D>::new(
            num_clusters,
            self.config.max_iteration,
            self.config.tolerance,
            self.config.num_features,
            KMeansVariant::Lloyd,
        );

        let flattened_dataset =
            self.get_sample_dataset_from_point_ids(&point_ids, num_points_for_clustering)?;
        let result = kmeans.fit(flattened_dataset)?;

        self.assign_docs_to_cluster(point_ids, result.centroids.as_ref())
    }

    fn compute_actual_num_clusters(
        &self,
        total_data_points: usize,
        num_clusters: usize,
        max_points_per_centroid: usize,
    ) -> usize {
        let num_centroids = num_clusters;
        let num_points_per_centroid = ceil_div(total_data_points, num_centroids);
        ceil_div(
            total_data_points,
            min(num_points_per_centroid, max_points_per_centroid),
        )
    }

    pub fn build_centroids(&mut self) -> Result<()> {
        debug!("Building centroids");

        // First pass to get the initial centroids
        let num_clusters = self.compute_actual_num_clusters(
            self.get_num_valid_vectors(),
            self.config.num_clusters,
            self.config.max_posting_list_size,
        );
        let kmeans = KMeansBuilder::<D>::new(
            num_clusters,
            self.config.max_iteration,
            self.config.tolerance,
            self.config.num_features,
            KMeansVariant::Lloyd,
        );

        // Sample the dataset to build the first set of centroids
        let mut rng = rand::thread_rng();

        // Create a vector from 0 to num_input_vectors and then shuffle it
        // Filter out invalidated ids
        let mut flattened_dataset: Vec<f32> = vec![];
        let indices: Vec<usize> = self
            .valid_point_id_mapping
            .values()
            .map(|&x| x as usize)
            .collect();

        let num_points_for_clustering =
            max(num_clusters, self.config.num_data_points_for_clustering);
        let selected = indices
            .choose_multiple(&mut rng, num_points_for_clustering)
            .cloned()
            .collect::<Vec<usize>>();
        selected.iter().for_each(|index| {
            flattened_dataset
                .extend_from_slice(self.vectors.get_no_context(*index as u32).unwrap());
        });

        let result = kmeans.fit(flattened_dataset)?;
        let posting_list_infos = self.assign_docs_to_cluster(indices, result.centroids.as_ref())?;

        // Repeatedly run kmeans on the longest posting list until no posting list is longer
        // than max_posting_list_size
        let mut heap = BinaryHeap::<PostingListInfo>::new();
        for posting_list_info in posting_list_infos {
            heap.push(posting_list_info);
        }

        let mut num_iter = 0;
        while !heap.is_empty() {
            match heap.peek() {
                None => break,
                Some(longest_posting_list) => {
                    if longest_posting_list.posting_list.len() <= self.config.max_posting_list_size
                    {
                        break;
                    }
                }
            }

            let longest_posting_list = heap.pop().unwrap();
            num_iter += 1;
            let new_posting_list_infos = self.cluster_docs(
                longest_posting_list.posting_list.clone(),
                self.config.max_posting_list_size,
            )?;

            // Add the new posting list infos to the heap
            for posting_list_info in new_posting_list_infos {
                heap.push(posting_list_info);
            }
        }
        debug!("Number of iterations to cluster: {}", num_iter);

        // Add the centroids to the centroid storage
        // We don't need to add the posting lists to the posting list storage, since later on
        // we will add them
        for posting_list_info in heap {
            if posting_list_info.posting_list.is_empty() {
                continue;
            }
            self.add_centroid(&posting_list_info.centroid)?;
        }

        Ok(())
    }

    pub fn build(&mut self) -> Result<()> {
        if self.get_num_valid_vectors() == 0 {
            return Err(anyhow!(
                "Trying to build an IVF index with empty vectors list"
            ));
        }
        self.build_centroids()?;
        self.build_posting_lists()?;

        Ok(())
    }

    fn build_posting_lists_with_stopping_points(
        &self,
    ) -> Result<Vec<PostingListWithStoppingPoints>> {
        let mut lists_with_stopping_points = Vec::new();
        let mut occurrence_map: HashMap<u64, Vec<usize>> = HashMap::new();

        for list_index in 0..self.posting_lists.len() {
            let posting_list = self.posting_lists.get(list_index as u32)?;
            lists_with_stopping_points.push(PostingListWithStoppingPoints::new(
                posting_list.iter().collect::<Vec<_>>(),
                SortedVec::new(),
            ));
            for vector_storage_index in posting_list.iter() {
                occurrence_map
                    .entry(vector_storage_index)
                    .or_default()
                    .push(list_index);
            }
        }

        for (stopping_point, posting_list_indices) in &occurrence_map {
            // Vector idx is not duplicated
            if posting_list_indices.len() == 1 {
                continue;
            }

            for list_index in posting_list_indices {
                lists_with_stopping_points[*list_index].add_stopping_point(*stopping_point);
            }
        }

        // Filter out the posting lists with non-empty stopping_points
        let filtered_lists: Vec<_> = lists_with_stopping_points
            .into_iter()
            .filter(|posting_list| !posting_list.stopping_points.is_empty())
            .collect();

        Ok(filtered_lists)
    }

    fn assign_ids_until_last_stopping_point(&mut self, assigned_ids: &mut [i64]) -> Result<i64> {
        let mut min_heap: BinaryHeap<Reverse<PostingListWithStoppingPoints>> = BinaryHeap::from(
            self.build_posting_lists_with_stopping_points()?
                .into_iter()
                .map(Reverse)
                .collect::<Vec<_>>(),
        );

        let mut cur_idx = 0;
        while let Some(Reverse(first_posting_list)) = min_heap.pop() {
            let min_dup_vec_idx = first_posting_list.stopping_points[0];

            // Collect all posting lists with the min stopping point
            let mut working_list = vec![first_posting_list];

            while let Some(Reverse(next_posting_list)) = min_heap.peek() {
                if next_posting_list.stopping_points[0] == min_dup_vec_idx {
                    working_list.push(min_heap.pop().unwrap().0); // Pop and unwrap safely
                } else {
                    break; // Exit if we reach a different stopping point
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
                            min_heap.push(Reverse(PostingListWithStoppingPoints::new(
                                list_with_stopping_points.posting_list[idx_in_posting_list + 1..]
                                    .to_vec(),
                                SortedVec::from_unsorted(
                                    list_with_stopping_points.stopping_points[1..].to_vec(),
                                ),
                            )));
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

        Ok(cur_idx)
    }

    /// Assign new ids to the vectors. The maximum number of vectors should be u32::MAX, however
    /// we want to use -1 as invalid id, hence the indices are i64.
    fn get_reassigned_ids(&mut self) -> Result<Vec<i64>> {
        let vector_length = self.vectors.num_vectors();
        let mut assigned_ids = vec![-1; vector_length];

        let mut cur_idx = self.assign_ids_until_last_stopping_point(&mut assigned_ids)?;

        for list_index in 0..self.posting_lists.len() {
            let posting_list = self.posting_lists.get(list_index as u32)?;
            for original_vector_index in posting_list.iter() {
                if assigned_ids[original_vector_index as usize] >= 0 {
                    continue;
                }
                assigned_ids[original_vector_index as usize] = cur_idx;
                cur_idx += 1;
            }
        }
        Ok(assigned_ids)
    }

    pub fn reassigned_mappings(&self) -> Option<&Vec<u32>> {
        self.reassigned_mappings.as_ref()
    }

    pub fn reindex(&mut self) -> Result<()> {
        let assigned_ids = self.get_reassigned_ids()?;
        self.reassigned_mappings = Some(assigned_ids.iter().map(|x| *x as u32).collect());

        // Update posting lists with reassigned IDs
        let new_posting_lists_path = format!(
            "{}/reindex/builder_posting_list_storage",
            self.config.base_directory
        );
        create_dir_all(&new_posting_lists_path)?;

        let mut new_posting_list_storage = FileBackedAppendablePostingListStorage::new(
            new_posting_lists_path,
            self.config.memory_size,
            self.config.file_size,
        );

        for list_index in 0..self.posting_lists.len() {
            let posting_list = self.posting_lists.get(list_index as u32)?;
            let mut new_posting_list = Vec::new();
            for original_vector_index in posting_list.iter() {
                new_posting_list.push(assigned_ids[original_vector_index as usize] as u64);
            }
            new_posting_list_storage.append(&new_posting_list)?;
        }
        self.posting_lists = new_posting_list_storage;

        // Update doc_id_mapping with reassigned IDs
        let tmp_id_provider = self.doc_id_mapping.clone();
        self.doc_id_mapping = vec![0; self.get_num_valid_vectors()];
        for (point_id, doc_id) in tmp_id_provider.into_iter().enumerate() {
            let new_point_id = assigned_ids.get(point_id).ok_or(anyhow!(
                "id in id_provider {} is larger than size of vectors",
                point_id
            ))?;
            // The doc_id originally mapped to this point_id has been invalidated
            // It can be revalidated though, so we can't check that the doc_id is
            // still invalid
            if *new_point_id == -1 {
                continue;
            }
            self.doc_id_mapping[*new_point_id as usize] = doc_id;
            assert!(self.valid_point_id_mapping.contains_key(&doc_id));
            self.valid_point_id_mapping
                .insert(doc_id, *new_point_id as u32);
        }

        // Build reverse assigned ids
        let mut reverse_assigned_ids = vec![-1; self.doc_id_mapping.len()];
        for (i, id) in assigned_ids.iter().enumerate() {
            if *id != -1 {
                reverse_assigned_ids[*id as usize] = i as i64;
            }
        }

        // Put the vectors to their reassigned places
        let new_vectors_path = format!(
            "{}/reindex/builder_vector_storage",
            self.config.base_directory
        );
        create_dir_all(&new_vectors_path)?;

        let mut new_vector_storage = FileBackedAppendableVectorStorage::<f32>::new(
            new_vectors_path,
            self.config.memory_size,
            self.config.file_size,
            self.config.num_features,
        );

        for mapped_id in reverse_assigned_ids.iter() {
            if *mapped_id != -1 {
                new_vector_storage
                    .append(self.vectors.get_no_context(*mapped_id as u32).unwrap())
                    .unwrap_or_else(|_| panic!("append failed"));
            }
        }

        self.vectors = new_vector_storage;
        Ok(())
    }

    pub fn cleanup(&mut self) -> Result<()> {
        let vectors_path = format!("{}/builder_vector_storage", self.config.base_directory);
        let centroids_path = format!("{}/builder_centroid_storage", self.config.base_directory);
        let posting_lists_path = format!(
            "{}/builder_posting_list_storage",
            self.config.base_directory
        );
        let reindex_path = format!("{}/reindex", self.config.base_directory);
        std::fs::remove_dir_all(&vectors_path)?;
        std::fs::remove_dir_all(&centroids_path)?;
        std::fs::remove_dir_all(&posting_lists_path)?;
        // It is ok to fail here, as we do not always have reindex path
        if let Err(err) = std::fs::remove_dir_all(&reindex_path) {
            if err.kind() != ErrorKind::NotFound {
                return Err(err.into());
            }
        }
        Ok(())
    }
}

// Test
#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use utils::test_utils::generate_random_vector;

    use super::*;

    fn count_files_with_prefix(directory: &Path, file_name_prefix: &str) -> usize {
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
        env_logger::try_init().ok();

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
        let mut builder: IvfBuilder<L2DistanceCalculator> = IvfBuilder::new(IvfBuilderConfig {
            max_iteration: 1000,
            batch_size: 4,
            num_clusters,
            num_data_points_for_clustering: num_vectors,
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
                .add_vector(i as u128, &[(i + 1) as f32])
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
        let mut builder: IvfBuilder<L2DistanceCalculator> = IvfBuilder::new(IvfBuilderConfig {
            max_iteration: 1000,
            batch_size: 4,
            num_clusters,
            num_data_points_for_clustering: num_vectors,
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

        assert!(builder.add_posting_list(&[1, 2, 3]).is_ok());
        assert!(builder.add_posting_list(&[4, 5, 6]).is_ok());
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
        let mut builder: IvfBuilder<L2DistanceCalculator> = IvfBuilder::new(IvfBuilderConfig {
            max_iteration: 1000,
            batch_size: 4,
            num_clusters,
            num_data_points_for_clustering: num_vectors,
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

        assert!(builder.add_posting_list(&[1, 2, 3]).is_ok());
        assert!(builder.add_posting_list(&[2, 4, 5]).is_ok());
        assert!(builder.add_posting_list(&[3, 6, 7]).is_ok());

        let result = builder.build_posting_lists_with_stopping_points().unwrap();

        // Expected result
        let expected_result = vec![
            PostingListWithStoppingPoints {
                posting_list: vec![1, 2, 3],
                stopping_points: SortedVec::from_unsorted(vec![2, 3]),
            },
            PostingListWithStoppingPoints {
                posting_list: vec![2, 4, 5],
                stopping_points: SortedVec::from_unsorted(vec![2]),
            },
            PostingListWithStoppingPoints {
                posting_list: vec![3, 6, 7],
                stopping_points: SortedVec::from_unsorted(vec![3]),
            },
        ];

        assert_eq!(result, expected_result);
    }

    #[test]
    fn test_assign_ids_until_last_stopping_point() {
        let temp_dir = tempdir::TempDir::new("assign_ids_until_last_stopping_point_test")
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
        let mut builder: IvfBuilder<L2DistanceCalculator> = IvfBuilder::new(IvfBuilderConfig {
            max_iteration: 1000,
            batch_size: 4,
            num_clusters,
            num_data_points_for_clustering: num_vectors,
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

        assert!(builder.add_posting_list(&[9, 18, 20]).is_ok());
        assert!(builder.add_posting_list(&[1, 3, 5, 7, 18, 20]).is_ok());
        assert!(builder.add_posting_list(&[14, 15, 16, 18]).is_ok());
        assert!(builder.add_posting_list(&[0, 2, 4, 6, 8, 20]).is_ok());
        assert!(builder.add_posting_list(&[10, 15, 21]).is_ok());

        let mut assigned_ids = vec![-1; 22];
        assert_eq!(
            builder
                .assign_ids_until_last_stopping_point(&mut assigned_ids)
                .expect("Failed to reassign ids for duplicated vectors"),
            16
        );

        assert_eq!(assigned_ids[10], 0);
        assert_eq!(assigned_ids[14], 1);
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
    fn test_get_reassigned_ids_0() {
        let temp_dir = tempdir::TempDir::new("get_reassigned_ids_0_test")
            .expect("Failed to create temporary directory");
        let base_directory = temp_dir
            .path()
            .to_str()
            .expect("Failed to convert temporary directory path to string")
            .to_string();
        let num_clusters = 4;
        let num_vectors = 22;
        let num_features = 1;
        let file_size = 4096 * 4096;
        let balance_factor = 0.0;
        let max_posting_list_size = usize::MAX;
        let mut builder: IvfBuilder<L2DistanceCalculator> = IvfBuilder::new(IvfBuilderConfig {
            max_iteration: 1000,
            batch_size: 4,
            num_clusters,
            num_data_points_for_clustering: num_vectors,
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

        for i in 0..num_vectors {
            builder
                .add_vector(i as u128, &generate_random_vector(num_features))
                .expect("Vector should be added");
        }

        assert!(builder.add_posting_list(&[11, 12, 13]).is_ok());
        assert!(builder.add_posting_list(&[0, 2, 4, 6, 8, 20]).is_ok());
        assert!(builder.add_posting_list(&[9, 18, 20]).is_ok());
        assert!(builder.add_posting_list(&[14, 15, 16, 18]).is_ok());
        assert!(builder.add_posting_list(&[1, 3, 5, 7, 18, 20]).is_ok());
        assert!(builder.add_posting_list(&[10, 15, 21]).is_ok());
        assert!(builder.add_posting_list(&[10, 15, 17, 19]).is_ok());

        let assigned_ids = builder
            .get_reassigned_ids()
            .expect("Failed to reassign ids");

        assert_eq!(assigned_ids[10], 0);
        assert_eq!(assigned_ids[14], 1);
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
        assert_eq!(assigned_ids[11], 16);
        assert_eq!(assigned_ids[12], 17);
        assert_eq!(assigned_ids[13], 18);
        assert_eq!(assigned_ids[21], 19);
        assert_eq!(assigned_ids[17], 20);
        assert_eq!(assigned_ids[19], 21);
    }

    #[test]
    fn test_get_reassigned_ids_1() {
        let temp_dir = tempdir::TempDir::new("get_reassigned_ids_1_test")
            .expect("Failed to create temporary directory");
        let base_directory = temp_dir
            .path()
            .to_str()
            .expect("Failed to convert temporary directory path to string")
            .to_string();
        let num_clusters = 4;
        let num_vectors = 22;
        let num_features = 1;
        let file_size = 4096 * 4096;
        let balance_factor = 0.0;
        let max_posting_list_size = usize::MAX;
        let mut builder: IvfBuilder<L2DistanceCalculator> = IvfBuilder::new(IvfBuilderConfig {
            max_iteration: 1000,
            batch_size: 4,
            num_clusters,
            num_data_points_for_clustering: num_vectors,
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

        for i in 0..num_vectors {
            builder
                .add_vector(i as u128, &generate_random_vector(num_features))
                .expect("Vector should be added");
        }

        assert!(builder.add_posting_list(&[0, 1, 2, 3]).is_ok());
        assert!(builder.add_posting_list(&[4, 5, 6, 7]).is_ok());
        assert!(builder.add_posting_list(&[8, 9, 10, 11]).is_ok());
        assert!(builder.add_posting_list(&[12, 13, 14, 15]).is_ok());
        assert!(builder.add_posting_list(&[16, 17, 18, 19]).is_ok());
        assert!(builder.add_posting_list(&[0, 4, 8, 12, 16]).is_ok());
        assert!(builder.add_posting_list(&[1, 5, 9, 13, 17]).is_ok());
        assert!(builder.add_posting_list(&[2, 6, 10, 14, 18]).is_ok());
        assert!(builder.add_posting_list(&[3, 7, 11, 15, 19]).is_ok());

        let assigned_ids = builder
            .get_reassigned_ids()
            .expect("Failed to reassign ids");

        assert_eq!(assigned_ids[0], 0);
        assert_eq!(assigned_ids[1], 1);
        assert_eq!(assigned_ids[2], 2);
        assert_eq!(assigned_ids[3], 3);
        assert_eq!(assigned_ids[4], 4);
        assert_eq!(assigned_ids[5], 5);
        assert_eq!(assigned_ids[6], 6);
        assert_eq!(assigned_ids[7], 7);
        assert_eq!(assigned_ids[8], 8);
        assert_eq!(assigned_ids[9], 9);
        assert_eq!(assigned_ids[10], 10);
        assert_eq!(assigned_ids[11], 11);
        assert_eq!(assigned_ids[12], 12);
        assert_eq!(assigned_ids[13], 13);
        assert_eq!(assigned_ids[14], 14);
        assert_eq!(assigned_ids[15], 15);
        assert_eq!(assigned_ids[16], 16);
        assert_eq!(assigned_ids[17], 17);
        assert_eq!(assigned_ids[18], 18);
        assert_eq!(assigned_ids[19], 19);
    }

    #[test]
    fn test_get_reassigned_ids_2() {
        let temp_dir = tempdir::TempDir::new("get_reassigned_ids_2_test")
            .expect("Failed to create temporary directory");
        let base_directory = temp_dir
            .path()
            .to_str()
            .expect("Failed to convert temporary directory path to string")
            .to_string();
        let num_clusters = 4;
        let num_vectors = 30;
        let num_features = 1;
        let file_size = 4096 * 4096;
        let balance_factor = 0.0;
        let max_posting_list_size = usize::MAX;
        let mut builder: IvfBuilder<L2DistanceCalculator> = IvfBuilder::new(IvfBuilderConfig {
            max_iteration: 1000,
            batch_size: 4,
            num_clusters,
            num_data_points_for_clustering: num_vectors,
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

        for i in 0..num_vectors {
            builder
                .add_vector(i as u128, &generate_random_vector(num_features))
                .expect("Vector should be added");
        }

        assert!(builder.add_posting_list(&[0, 5, 10, 15, 20, 25]).is_ok());
        assert!(builder.add_posting_list(&[1, 6, 11, 16, 21, 26]).is_ok());
        assert!(builder.add_posting_list(&[0, 7, 12, 17, 22, 27]).is_ok());
        assert!(builder.add_posting_list(&[2, 8, 13, 18, 23, 28]).is_ok());
        assert!(builder.add_posting_list(&[3, 9, 14, 19, 24, 29]).is_ok());
        assert!(builder.add_posting_list(&[4, 20, 21, 22, 23, 24]).is_ok());
        assert!(builder.add_posting_list(&[1, 25, 26, 27, 28, 29]).is_ok());

        let assigned_ids = builder
            .get_reassigned_ids()
            .expect("Failed to reassign ids");

        assert_eq!(assigned_ids[0], 0);
        assert_eq!(assigned_ids[1], 1);
        assert_eq!(assigned_ids[4], 2);
        assert_eq!(assigned_ids[5], 3);
        assert_eq!(assigned_ids[10], 4);
        assert_eq!(assigned_ids[15], 5);
        assert_eq!(assigned_ids[20], 6);
        assert_eq!(assigned_ids[6], 7);
        assert_eq!(assigned_ids[11], 8);
        assert_eq!(assigned_ids[16], 9);
        assert_eq!(assigned_ids[21], 10);
        assert_eq!(assigned_ids[7], 11);
        assert_eq!(assigned_ids[12], 12);
        assert_eq!(assigned_ids[17], 13);
        assert_eq!(assigned_ids[22], 14);
        assert_eq!(assigned_ids[2], 15);
        assert_eq!(assigned_ids[8], 16);
        assert_eq!(assigned_ids[13], 17);
        assert_eq!(assigned_ids[18], 18);
        assert_eq!(assigned_ids[23], 19);
        assert_eq!(assigned_ids[3], 20);
        assert_eq!(assigned_ids[9], 21);
        assert_eq!(assigned_ids[14], 22);
        assert_eq!(assigned_ids[19], 23);
        assert_eq!(assigned_ids[24], 24);
        assert_eq!(assigned_ids[25], 25);
        assert_eq!(assigned_ids[26], 26);
        assert_eq!(assigned_ids[27], 27);
        assert_eq!(assigned_ids[28], 28);
        assert_eq!(assigned_ids[29], 29);
    }

    #[test]
    fn test_get_reassigned_ids_3() {
        let temp_dir = tempdir::TempDir::new("get_reassigned_ids_3_test")
            .expect("Failed to create temporary directory");
        let base_directory = temp_dir
            .path()
            .to_str()
            .expect("Failed to convert temporary directory path to string")
            .to_string();
        let num_clusters = 4;
        let num_vectors = 30;
        let num_features = 1;
        let file_size = 4096 * 4096;
        let balance_factor = 0.0;
        let max_posting_list_size = usize::MAX;
        let mut builder: IvfBuilder<L2DistanceCalculator> = IvfBuilder::new(IvfBuilderConfig {
            max_iteration: 1000,
            batch_size: 4,
            num_clusters,
            num_data_points_for_clustering: num_vectors,
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

        for i in 0..num_vectors {
            builder
                .add_vector(i as u128, &generate_random_vector(num_features))
                .expect("Vector should be added");
        }

        assert!(builder.add_posting_list(&[0, 4, 8, 12, 16, 20]).is_ok());
        assert!(builder.add_posting_list(&[1, 5, 9, 13, 17, 21]).is_ok());
        assert!(builder.add_posting_list(&[2, 6, 10, 14, 18, 22]).is_ok());
        assert!(builder.add_posting_list(&[3, 7, 11, 15, 19, 23]).is_ok());
        assert!(builder.add_posting_list(&[0, 6, 12, 18]).is_ok());
        assert!(builder.add_posting_list(&[1, 7, 13, 19]).is_ok());

        let assigned_ids = builder
            .get_reassigned_ids()
            .expect("Failed to reassign ids");

        assert_eq!(assigned_ids[0], 0);
        assert_eq!(assigned_ids[1], 1);
        assert_eq!(assigned_ids[2], 2);
        assert_eq!(assigned_ids[6], 3);
        assert_eq!(assigned_ids[3], 4);
        assert_eq!(assigned_ids[7], 5);
        assert_eq!(assigned_ids[4], 6);
        assert_eq!(assigned_ids[8], 7);
        assert_eq!(assigned_ids[12], 8);
        assert_eq!(assigned_ids[5], 9);
        assert_eq!(assigned_ids[9], 10);
        assert_eq!(assigned_ids[13], 11);
        assert_eq!(assigned_ids[10], 12);
        assert_eq!(assigned_ids[14], 13);
        assert_eq!(assigned_ids[18], 14);
        assert_eq!(assigned_ids[11], 15);
        assert_eq!(assigned_ids[15], 16);
        assert_eq!(assigned_ids[19], 17);
        assert_eq!(assigned_ids[16], 18);
        assert_eq!(assigned_ids[20], 19);
        assert_eq!(assigned_ids[17], 20);
        assert_eq!(assigned_ids[21], 21);
        assert_eq!(assigned_ids[22], 22);
        assert_eq!(assigned_ids[23], 23);
    }

    #[test]
    fn test_ivf_builder_reindex() {
        let temp_dir = tempdir::TempDir::new("ivf_builder_reindex_test")
            .expect("Failed to create temporary directory");
        let base_directory = temp_dir
            .path()
            .to_str()
            .expect("Failed to convert temporary directory path to string")
            .to_string();
        let num_clusters = 4;
        let num_features = 1;
        let file_size = 4096;
        let balance_factor = 0.0;
        let max_posting_list_size = usize::MAX;
        const NUM_VECTORS: usize = 22;
        let mut builder: IvfBuilder<L2DistanceCalculator> = IvfBuilder::new(IvfBuilderConfig {
            max_iteration: 1000,
            batch_size: 4,
            num_clusters,
            num_data_points_for_clustering: NUM_VECTORS,
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

        for i in 0..NUM_VECTORS {
            builder
                .add_vector(i as u128 + 100, &[i as f32])
                .expect("Vector should be added");
        }

        assert!(builder.add_posting_list(&[11, 12, 13]).is_ok());
        assert!(builder.add_posting_list(&[0, 2, 4, 6, 8, 20]).is_ok());
        assert!(builder.add_posting_list(&[9, 18, 20]).is_ok());
        assert!(builder.add_posting_list(&[14, 15, 16, 18]).is_ok());
        assert!(builder.add_posting_list(&[1, 3, 5, 7, 18, 20]).is_ok());
        assert!(builder.add_posting_list(&[10, 15, 21]).is_ok());
        assert!(builder.add_posting_list(&[10, 15, 17, 19]).is_ok());

        builder.reindex().expect("Failed to reindex");

        let expected_vectors: [f32; NUM_VECTORS] = [
            10.0, 14.0, 15.0, 1.0, 3.0, 5.0, 7.0, 9.0, 16.0, 18.0, 0.0, 2.0, 4.0, 6.0, 8.0, 20.0,
            11.0, 12.0, 13.0, 21.0, 17.0, 19.0,
        ];

        for (i, expected_vector) in expected_vectors.iter().enumerate().take(NUM_VECTORS) {
            assert_eq!(
                builder
                    .vectors
                    .get_no_context(i as u32)
                    .unwrap_or_else(|_| panic!("Failed to get vector at index {i}"))[0],
                *expected_vector
            );
        }

        let expected_doc_ids: [u128; NUM_VECTORS] = [
            10, 14, 15, 1, 3, 5, 7, 9, 16, 18, 0, 2, 4, 6, 8, 20, 11, 12, 13, 21, 17, 19,
        ];

        assert_eq!(builder.doc_id_mapping.len(), NUM_VECTORS);

        for (expected_doc_id, doc_id) in expected_doc_ids.iter().zip(builder.doc_id_mapping.iter())
        {
            assert_eq!(*doc_id, expected_doc_id + 100);
        }
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
        let mut builder: IvfBuilder<L2DistanceCalculator> = IvfBuilder::new(IvfBuilderConfig {
            max_iteration: 1000,
            batch_size: 4,
            num_clusters,
            num_data_points_for_clustering: num_vectors,
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
                .add_vector(i as u128, &generate_random_vector(num_features))
                .expect("Vector should be added");
        }

        let result = builder.build();
        assert!(result.is_ok());

        assert_eq!(builder.vectors.num_vectors(), num_vectors);
        assert_eq!(builder.centroids.num_vectors(), num_clusters);
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

    #[test]
    fn test_ivf_builder_with_invalidated() {
        let temp_dir = tempdir::TempDir::new("ivf_builder_with_invalidated_test")
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
        let mut builder: IvfBuilder<L2DistanceCalculator> = IvfBuilder::new(IvfBuilderConfig {
            max_iteration: 1000,
            batch_size: 4,
            num_clusters,
            num_data_points_for_clustering: num_vectors,
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
                .add_vector(i as u128, &generate_random_vector(num_features))
                .expect("Vector should be added");
        }
        for i in 0..num_vectors {
            if i % 2 == 0 {
                assert!(builder.invalidate(i as u128));
            }
        }

        let result = builder.build();
        assert!(result.is_ok());

        assert_eq!(builder.get_num_valid_vectors(), num_vectors / 2);
        assert_eq!(builder.centroids.num_vectors(), num_clusters);
        assert_eq!(builder.posting_lists.len(), num_clusters);

        assert!(builder.reindex().is_ok());
        for i in 0..num_clusters {
            let posting_list = builder
                .posting_lists()
                .get(i as u32)
                .expect("Failed to get a posting list");
            for index in posting_list.iter() {
                assert!(index < (num_vectors / 2) as u64);
            }
        }
    }

    #[test]
    fn test_ivf_builder_with_revalidated() {
        let temp_dir = tempdir::TempDir::new("ivf_builder_with_revalidated_test")
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

        let mut builder: IvfBuilder<L2DistanceCalculator> = IvfBuilder::new(IvfBuilderConfig {
            max_iteration: 1000,
            batch_size: 4,
            num_clusters,
            num_data_points_for_clustering: num_vectors,
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
        for i in 0..num_vectors {
            builder
                .add_vector(i as u128, &generate_random_vector(num_features))
                .expect("Vector should be added");
        }
        for i in 0..num_vectors - 1 {
            assert!(builder.invalidate(i as u128));
        }
        // Test revalidation
        for i in 0..num_vectors - 1 {
            builder
                .add_vector(i as u128, &generate_random_vector(num_features))
                .expect("Vector should be added");
        }

        // Test doc_id reassignment
        assert!(builder
            .valid_point_id_mapping
            .values()
            .any(|&valid_point_id| valid_point_id == (num_vectors - 1) as u32));
        assert!(builder
            .add_vector(
                (num_vectors - 1) as u128,
                &generate_random_vector(num_features)
            )
            .is_ok());
        assert!(!builder
            .valid_point_id_mapping
            .values()
            .any(|&valid_point_id| valid_point_id == (num_vectors - 1) as u32));
        assert!(builder
            .valid_point_id_mapping
            .values()
            .any(|&valid_point_id| valid_point_id == (num_vectors * 2 - 1) as u32));

        let result = builder.build();
        assert!(result.is_ok());

        assert_eq!(builder.get_num_valid_vectors(), num_vectors);
        assert_eq!(builder.centroids.num_vectors(), num_clusters);
        assert_eq!(builder.posting_lists.len(), num_clusters);

        assert!(builder.reindex().is_ok());
        for i in 0..num_clusters {
            let posting_list = builder
                .posting_lists()
                .get(i as u32)
                .expect("Failed to get a posting list");
            for index in posting_list.iter() {
                assert!(index < num_vectors as u64);
            }
        }
    }

    #[test]
    fn test_ivf_builder_reindex_with_invalidated() {
        let temp_dir = tempdir::TempDir::new("ivf_builder_with_invalidated_test")
            .expect("Failed to create temporary directory");
        let base_directory = temp_dir
            .path()
            .to_str()
            .expect("Failed to convert temporary directory path to string")
            .to_string();
        let num_clusters = 1;
        let num_vectors = 1000;
        let num_features = 4;
        let file_size = 4096;
        let balance_factor = 0.0;
        let max_posting_list_size = usize::MAX;
        let mut builder: IvfBuilder<L2DistanceCalculator> = IvfBuilder::new(IvfBuilderConfig {
            max_iteration: 1000,
            batch_size: 4,
            num_clusters,
            num_data_points_for_clustering: num_vectors,
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
                .add_vector(i as u128, &generate_random_vector(num_features))
                .expect("Vector should be added");
        }
        for i in 0..num_vectors {
            if i % 2 == 0 {
                assert!(builder.invalidate(i as u128));
            }
        }

        let result = builder.build();
        assert!(result.is_ok());

        assert_eq!(builder.get_num_valid_vectors(), num_vectors / 2);

        assert!(builder.reindex().is_ok());
        let posting_list = builder
            .posting_lists()
            .get(0)
            .expect("Failed to get a posting list");
        assert_eq!(posting_list.elem_count, num_vectors / 2);
        for (i, index) in posting_list.iter().enumerate() {
            assert!(index == i as u64);
        }
    }

    #[test]
    fn test_sample() {
        let num: Vec<usize> = (0..100).collect();
        let mut rng = rand::thread_rng();
        let sample = num
            .choose_multiple(&mut rng, 10)
            .cloned()
            .collect::<Vec<usize>>();
        println!("{:?}", sample);
    }
}
