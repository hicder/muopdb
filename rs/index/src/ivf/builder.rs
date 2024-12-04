use std::collections::BinaryHeap;
use std::fs::{create_dir, create_dir_all};

use anyhow::Result;
use rand::seq::SliceRandom;
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

    // Parameters for storages.
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
            config.num_clusters,
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
    ) -> Result<Vec<usize>> {
        let mut distances: Vec<(usize, f32)> = Vec::new();
        let num_centroids = centroids.len();
        for i in 0..num_centroids {
            let centroid = centroids.get(i as u32)?;
            let dist = L2DistanceCalculator::calculate_squared(&vector, &centroid);
            if dist.is_nan() {
                println!("NAN found");
            }
            distances.push((i, dist));
        }
        distances.select_nth_unstable_by(num_probes - 1, |a, b| a.1.total_cmp(&b.1));
        distances.truncate(num_probes);
        Ok(distances.into_iter().map(|(idx, _)| idx).collect())
    }

    pub fn build_posting_lists(&mut self) -> Result<()> {
        let mut posting_lists: Vec<Vec<u64>> = vec![Vec::with_capacity(0); self.centroids.len()];
        // Assign vectors to nearest centroids
        for i in 0..self.vectors.len() {
            let vector = self.vectors.get(i as u32)?;
            let nearest_centroid = Self::find_nearest_centroids(
                &vector,
                self.centroids.as_ref(),
                self.config.max_clusters_per_vector,
            )?;

            for centroid_id in nearest_centroid {
                posting_lists[centroid_id].push(i as u64);
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
            self.centroids.len(),
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

    /// Train kmeans on the dataset, and generate the posting lists
    pub fn build(&mut self) -> Result<()> {
        self.build_centroids()?;
        self.build_posting_lists()?;

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
    fn test_ivf_builder() {
        env_logger::init();

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
