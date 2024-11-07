use kmeans::*;
use log::debug;

use crate::index::Index;
use crate::ivf::index::Ivf;

pub struct IvfBuilderConfig {
    pub max_iteration: usize,
    pub batch_size: usize,
    pub num_clusters: usize,
    pub num_probes: usize,
}

pub struct IvfBuilder {
    config: IvfBuilderConfig,
    dataset: Vec<Vec<f32>>,
}

impl IvfBuilder {
    /// Create a new IvfBuilder
    pub fn new(config: IvfBuilderConfig) -> Self {
        Self {
            config,
            dataset: Vec::new(),
        }
    }

    /// Add a new vector to the dataset for training
    pub fn add(&mut self, data: Vec<f32>) {
        self.dataset.push(data);
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
        let flattened_dataset: Vec<f32> = self.dataset.iter().flatten().cloned().collect();
        let dim = self.dataset[0].len();
        let kmeans: KMeans<_, 8> = KMeans::new(flattened_dataset, self.dataset.len(), dim);
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
            .chunks(dim)
            .map(|chunk| chunk.to_vec())
            .collect();
        let inverted_lists = Ivf::build_inverted_lists(&self.dataset, &centroids);
        Ivf::new(
            self.dataset.clone(),
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
    use crate::utils::SearchContext;

    #[test]
    fn test_ivf_builder() {
        env_logger::init();

        const DIMENSION: usize = 128;
        let mut builder = IvfBuilder::new(IvfBuilderConfig {
            max_iteration: 1000,
            batch_size: 4,
            num_clusters: 10,
            num_probes: 3,
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
