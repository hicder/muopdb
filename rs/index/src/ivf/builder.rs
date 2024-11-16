use std::collections::HashMap;
use std::fs::File;
use std::io::BufWriter;

use kmeans::*;
use log::debug;

use crate::ivf::index::Ivf;
use crate::utils::SearchContext;
use crate::vector::file::FileBackedAppendableVectorStorage;
use crate::vector::fixed_file::FixedFileVectorStorage;
use crate::vector::VectorStorage;

pub struct IvfBuilderConfig {
    pub max_iteration: usize,
    pub batch_size: usize,
    pub num_clusters: usize,
    pub num_probes: usize,
    // Parameters for vector storage.
    pub base_directory: String,
    pub vector_storage_memory_size: usize,
    pub vector_storage_file_size: usize,
    pub num_features: usize,
}

pub struct IvfBuilder {
    config: IvfBuilderConfig,
    vectors: Box<dyn VectorStorage<f32>>,
}

impl IvfBuilder {
    /// Create a new IvfBuilder
    pub fn new(config: IvfBuilderConfig) -> Self {
        let vectors = Box::new(FileBackedAppendableVectorStorage::<f32>::new(
            config.base_directory.clone(),
            config.vector_storage_memory_size,
            config.vector_storage_file_size,
            config.num_features,
        ));
        Self { config, vectors }
    }

    /// Add a new vector to the dataset for training
    pub fn add(&mut self, data: Vec<f32>) {
        self.vectors
            .append(&data)
            .unwrap_or_else(|_| panic!("append to Ivf failed"));
    }

    pub fn build_inverted_lists(
        vector_storage: &FixedFileVectorStorage<f32>,
        centroids: &Vec<Vec<f32>>,
    ) -> HashMap<usize, Vec<usize>> {
        let mut context = SearchContext::new(false);
        let mut inverted_lists = HashMap::new();
        // Assign vectors to nearest centroid
        for i in 0..vector_storage.num_vectors {
            let vector = vector_storage.get(i, &mut context).unwrap().to_vec();
            let nearest_centroid = Ivf::find_nearest_centroids(&vector, &centroids, 1);
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

        let centroids = result
            .centroids
            .chunks(self.config.num_features)
            .map(|chunk| chunk.to_vec())
            .collect();

        let vectors_path = format!("{}/immutable_vector_storage", self.config.base_directory);
        let mut vectors_file = File::create(vectors_path.clone()).unwrap();
        let mut vectors_buffer_writer = BufWriter::new(&mut vectors_file);

        self.vectors.write(&mut vectors_buffer_writer).unwrap();

        let storage =
            FixedFileVectorStorage::<f32>::new(vectors_path, self.config.num_features).unwrap();
        let inverted_lists = Self::build_inverted_lists(&storage, &centroids);
        Ivf::new(
            storage,
            centroids,
            inverted_lists,
            self.config.num_clusters,
            self.config.num_probes,
        )
    }
}

// Test
#[cfg(test)]
mod tests {
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
            builder.add(generate_random_vector(DIMENSION));
        }

        let index = builder.build();

        let query = generate_random_vector(DIMENSION);
        let mut context = SearchContext::new(false);
        let results = index.search(&query, 5, 0, &mut context);
        match results {
            Some(results) => {
                assert_eq!(results.len(), 5);
                // Make sure results are in ascending order
                assert!(results.windows(2).all(|w| w[0] <= w[1]));
            }
            None => {
                assert!(false);
            }
        }
    }
}
