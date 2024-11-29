use std::collections::HashSet;
use std::fs::{create_dir, create_dir_all};

use anyhow::Result;
use kmeans::*;
use log::debug;
use rand::Rng;
use utils::distance::l2::L2DistanceCalculator;
use utils::DistanceCalculator;

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
}

pub struct IvfBuilder {
    config: IvfBuilderConfig,
    vectors: Box<dyn VectorStorage<f32>>,
    centroids: Box<dyn VectorStorage<f32>>,
    posting_lists: Box<dyn for<'a> PostingListStorage<'a>>,
}

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
        })
    }

    pub fn config(&self) -> &IvfBuilderConfig {
        &self.config
    }

    pub fn vectors(&self) -> &dyn VectorStorage<f32> {
        &*self.vectors
    }

    pub fn centroids(&self) -> &dyn VectorStorage<f32> {
        &*self.centroids
    }

    pub fn posting_lists_mut(&mut self) -> &mut dyn for<'a> PostingListStorage<'a> {
        &mut *self.posting_lists
    }

    /// Add a new vector to the dataset for training
    pub fn add_vector(&mut self, data: Vec<f32>) -> Result<()> {
        self.vectors.append(&data)?;
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

    fn find_nearest_centroids(
        vector: &[f32],
        centroids: &dyn VectorStorage<f32>,
        num_probes: usize,
    ) -> Result<Vec<usize>> {
        let mut distances: Vec<(usize, f32)> = Vec::new();
        for i in 0..centroids.len() {
            let centroid = centroids.get(i as u32)?;
            let dist = L2DistanceCalculator::calculate(&vector, &centroid);
            distances.push((i, dist));
        }
        distances.select_nth_unstable_by(num_probes - 1, |a, b| a.1.total_cmp(&b.1));
        Ok(distances.into_iter().map(|(idx, _)| idx).collect())
    }

    pub fn build_posting_lists(&mut self) -> Result<()> {
        let mut posting_lists: Vec<Vec<u64>> =
            vec![Vec::with_capacity(0); self.config.num_clusters as usize];
        // Assign vectors to nearest centroid
        for i in 0..self.vectors.len() {
            let vector = self.vectors.get(i as u32)?;
            let nearest_centroid = Self::find_nearest_centroids(
                &vector,
                self.centroids.as_ref(),
                self.config.max_clusters_per_vector,
            )?;
            posting_lists[nearest_centroid[0]].push(i as u64);
        }

        // Move ownership of each posting list to the posting list storage
        for posting_list in posting_lists.into_iter() {
            self.add_posting_list(posting_list.as_ref())?;
        }
        Ok(())
    }

    /// Train kmeans on the dataset, and generate the posting lists
    pub fn build(&mut self) -> Result<()> {
        let config = KMeansConfig::build()
            .init_done(&|_| debug!("Initialization completed."))
            .iteration_done(&|s, nr, new_distsum| {
                debug!(
                    "Iteration {} - Error: {:.2} -> {:.2} | Improvement: {:.2}",
                    nr,
                    s.distsum,
                    new_distsum,
                    s.distsum - new_distsum
                )
            })
            .build();
        let mut rng = rand::thread_rng();
        let num_vectors = self.config.num_data_points;
        let mut unique_indices = HashSet::with_capacity(num_vectors);
        let mut flattened_dataset = Vec::new();

        while unique_indices.len() < num_vectors {
            let random_index = rng.gen_range(0..num_vectors);
            if unique_indices.insert(random_index) {
                let vector = self.vectors.get(random_index as u32)?;
                flattened_dataset.extend_from_slice(vector);
            }
        }

        let kmeans: KMeans<_, 8> =
            KMeans::new(flattened_dataset, num_vectors, self.config.num_features);
        let result = kmeans.kmeans_minibatch(
            self.config.batch_size,
            self.config.num_clusters,
            self.config.max_iteration,
            KMeans::init_random_sample,
            &config,
        );
        debug!("Error: {}", result.distsum);

        let centroids_iter = result.centroids.chunks(self.config.num_features);
        for centroid in centroids_iter {
            self.add_centroid(centroid)?;
        }

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
        })
        .expect("Failed to create builder");
        // Generate 1000 vectors of f32, dimension 4
        for _ in 0..num_vectors {
            builder
                .add_vector(generate_random_vector(num_features))
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
        let posting_lists_path =
            PathBuf::from(&builder.config.base_directory).join("builder_posting_list_storage");
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
