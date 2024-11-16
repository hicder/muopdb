use std::collections::HashMap;
use std::fs::{create_dir, File};
use std::io::BufWriter;

use kmeans::*;
use log::debug;

use crate::ivf::index::Ivf;
use crate::utils::SearchContext;
use crate::vector::file::FileBackedAppendableVectorStorage;
use crate::vector::fixed_file::FixedFileVectorStorage;
use crate::vector::{VectorStorage, VectorStorageConfig};

pub struct IvfBuilderConfig {
    pub max_iteration: usize,
    pub batch_size: usize,
    pub num_clusters: usize,
    pub num_probes: usize,
    // Parameters for vector and centroid storages.
    pub base_directory: String,
    pub vector_storage_memory_size: usize,
    pub vector_storage_file_size: usize,
    pub num_features: usize,
}

pub struct IvfBuilder {
    config: IvfBuilderConfig,
    vectors: Box<dyn VectorStorage<f32>>,
    centroids: Box<dyn VectorStorage<f32>>,
}

impl IvfBuilder {
    /// Create a new IvfBuilder
    pub fn new(config: IvfBuilderConfig) -> Self {
        let vectors_path = format!("{}/dataset", config.base_directory);
        create_dir(vectors_path.clone());
        let vectors = Box::new(FileBackedAppendableVectorStorage::<f32>::new(
            vectors_path,
            config.vector_storage_memory_size,
            config.vector_storage_file_size,
            config.num_features,
        ));
        let centroids_path = format!("{}/centroids", config.base_directory);
        create_dir(centroids_path.clone());
        let centroids = Box::new(FileBackedAppendableVectorStorage::<f32>::new(
            centroids_path,
            config.vector_storage_memory_size,
            config.vector_storage_file_size,
            config.num_features,
        ));
        Self {
            config,
            vectors,
            centroids,
        }
    }

    /// Add a new vector to the dataset for training
    pub fn add_vector(&mut self, data: Vec<f32>) {
        self.vectors
            .append(&data)
            .unwrap_or_else(|_| panic!("append to Ivf failed"));
    }

    /// Add a new centroid
    pub fn add_centroid(&mut self, centroid: &[f32]) {
        self.centroids
            .append(centroid)
            .unwrap_or_else(|_| panic!("append to Ivf failed"));
    }

    pub fn build_inverted_lists(
        vector_storage: &FixedFileVectorStorage<f32>,
        centroid_storage: &FixedFileVectorStorage<f32>,
    ) -> HashMap<usize, Vec<usize>> {
        let mut context = SearchContext::new(false);
        let mut inverted_lists = HashMap::new();
        // Assign vectors to nearest centroid
        for i in 0..vector_storage.num_vectors {
            let vector = vector_storage.get(i, &mut context).unwrap().to_vec();
            let nearest_centroid = Ivf::find_nearest_centroids(&vector, &centroid_storage, 1);
            inverted_lists
                .entry(nearest_centroid[0])
                .or_insert_with(Vec::new)
                .push(i);
        }
        inverted_lists
    }

    /// Train kmeans on the dataset, and returns the Ivf index
    pub fn build(&mut self) -> Ivf {
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
        let mut flattened_dataset: Vec<f32> = Vec::new();
        let num_vectors = self.vectors.len();
        for i in 0..num_vectors {
            if let Some(vector) = self.vectors.get(i as u32) {
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

        let vectors_path = format!("{}/immutable_vector_storage", self.config.base_directory);
        let mut vectors_file = File::create(vectors_path.clone()).unwrap();
        let mut vectors_buffer_writer = BufWriter::new(&mut vectors_file);

        self.vectors.write(&mut vectors_buffer_writer).unwrap();

        let mut context = SearchContext::new(false);
        let storage =
            FixedFileVectorStorage::<f32>::new(vectors_path, self.config.num_features).unwrap();

        let centroids_iter = result.centroids.chunks(self.config.num_features);
        for centroid in centroids_iter {
            self.add_centroid(centroid);
        }
        let centroids_path = format!("{}/immutable_centroid_storage", self.config.base_directory);
        let mut centroids_file = File::create(centroids_path.clone()).unwrap();
        let mut centroids_buffer_writer = BufWriter::new(&mut centroids_file);

        self.centroids.write(&mut centroids_buffer_writer).unwrap();

        let centroid_storage =
            FixedFileVectorStorage::<f32>::new(centroids_path, self.config.num_features).unwrap();

        let inverted_lists = Self::build_inverted_lists(&storage, &centroid_storage);
        Ivf::new(
            storage,
            centroid_storage,
            inverted_lists,
            self.config.num_clusters,
            self.config.num_probes,
        )
    }
}

// Test
#[cfg(test)]
mod tests {
    use std::path::Path;

    use utils::test_utils::generate_random_vector;

    use super::*;
    use crate::index::Index;
    use crate::utils::SearchContext;

    #[test]
    fn test_ivf_builder() {
        env_logger::init();

        const DIMENSION: usize = 128;
        let temp_dir = tempdir::TempDir::new("ivf_builder_test").unwrap();
        let base_directory = temp_dir.path().to_str().unwrap().to_string();
        let mut builder = IvfBuilder::new(IvfBuilderConfig {
            max_iteration: 1000,
            batch_size: 4,
            num_clusters: 10,
            num_probes: 3,
            base_directory,
            vector_storage_memory_size: 1024,
            vector_storage_file_size: 4096,
            num_features: DIMENSION,
        });
        // Generate 10000 vectors of f32, dimension 128
        for _ in 0..10000 {
            builder.add_vector(generate_random_vector(DIMENSION));
        }

        //  let index = builder.build();

        //  let query = generate_random_vector(DIMENSION);
        //  let mut context = SearchContext::new(false);
        //  let results = index.search(&query, 5, 0, &mut context);
        //  match results {
        //      Some(results) => {
        //          assert_eq!(results.len(), 5);
        //          // Make sure results are in ascending order
        //          assert!(results.windows(2).all(|w| w[0] <= w[1]));
        //      }
        //      None => {
        //          assert!(false);
        //      }
        //  }
    }
}
