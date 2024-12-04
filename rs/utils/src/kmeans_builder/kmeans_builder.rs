use anyhow::{anyhow, Ok, Result};
use kmeans::KMeansConfig;
use log::debug;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use rayon::slice::ParallelSlice;

use crate::distance::l2::{L2DistanceCalculator, LaneConformingL2DistanceCalculator};
use crate::CalculateSquared;

#[derive(PartialEq, Debug)]
pub enum KMeansVariant {
    Lloyd,
}

pub struct KMeansBuilder {
    pub num_cluters: usize,
    pub max_iter: usize,

    // Factor which determine how much penalty large cluster has over small cluster.
    pub tolerance: f32,

    // data shape
    pub dimension: usize,

    // Variant for this algorithm. Currently only Lloyd is supported.
    pub variant: KMeansVariant,

    pub cluster_init_values: Option<Vec<usize>>,
}

pub struct KMeansResult {
    // Flattened centroids
    pub centroids: Vec<f32>,
    pub assignments: Vec<usize>,
}

// TODO(hicder): Add support for different variants of k-means.
// TODO(hicder): Add support for different distance metrics.
impl KMeansBuilder {
    pub fn new(
        num_cluters: usize,
        max_iter: usize,
        tolerance: f32,
        dimension: usize,
        variant: KMeansVariant,
    ) -> Self {
        Self {
            num_cluters,
            max_iter,
            tolerance,
            dimension,
            variant,
            cluster_init_values: None,
        }
    }

    pub fn new_with_cluster_init_values(
        num_cluters: usize,
        max_iter: usize,
        tolerance: f32,
        dimension: usize,
        variant: KMeansVariant,
        cluster_init_values: Vec<usize>,
    ) -> Self {
        Self {
            num_cluters,
            max_iter,
            tolerance,
            dimension,
            variant,
            cluster_init_values: Some(cluster_init_values),
        }
    }

    // Use as part of benchmarking. Don't use in production.
    pub fn fit_old(&self, data: Vec<f32>) -> Result<KMeansResult> {
        let sample_count = data.len() / self.dimension;
        let conf = KMeansConfig::build()
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
        let kmean: kmeans::KMeans<_, 16> = kmeans::KMeans::new(data, sample_count, self.dimension);
        let result = kmean.kmeans_lloyd(
            self.num_cluters,
            self.max_iter,
            kmeans::KMeans::init_random_sample,
            &conf,
        );
        let kmeans_result = KMeansResult {
            centroids: result.centroids,
            assignments: result.assignments,
        };
        Ok(kmeans_result)
    }

    pub fn fit(&self, flattened_data: Vec<f32>) -> Result<KMeansResult> {
        // Validate dimension
        if flattened_data.len() % self.dimension != 0 {
            return Err(anyhow!(
                "Dimension of data point {} is not equal to dimension of KMeans object {}",
                flattened_data.len(),
                self.dimension
            )); // TODO(hicder): Better error message
        }

        match self.variant {
            KMeansVariant::Lloyd => {
                if self.dimension % 16 == 0 {
                    return self
                        .run_lloyd::<LaneConformingL2DistanceCalculator<16>>(flattened_data);
                } else if self.dimension % 8 == 0 {
                    return self.run_lloyd::<LaneConformingL2DistanceCalculator<8>>(flattened_data);
                } else if self.dimension % 4 == 0 {
                    return self.run_lloyd::<LaneConformingL2DistanceCalculator<4>>(flattened_data);
                } else {
                    return self.run_lloyd::<L2DistanceCalculator>(flattened_data);
                }
            }
        }
    }

    fn run_lloyd<T: CalculateSquared + Send + Sync>(
        &self,
        flattened_data_points: Vec<f32>,
    ) -> Result<KMeansResult> {
        let data_points = flattened_data_points
            .par_chunks_exact(self.dimension)
            .map(|x| x)
            .collect::<Vec<&[f32]>>();

        let num_data_points = data_points.len();
        let mut cluster_labels = vec![0; num_data_points];

        // Random initialization of cluster labels
        match &self.cluster_init_values {
            Some(values) => {
                for i in 0..num_data_points {
                    cluster_labels[i] = values[i % values.len()];
                }
            }
            None => {
                for i in 0..num_data_points {
                    cluster_labels[i] = rand::random::<usize>() % self.num_cluters;
                }
            }
        }

        let mut cluster_sizes = vec![0; self.num_cluters];
        for i in 0..num_data_points {
            cluster_sizes[cluster_labels[i]] += 1;
        }

        let mut centroids = vec![0.0; self.num_cluters * self.dimension];
        for i in 0..num_data_points {
            let data_point = &data_points[i];
            let label = cluster_labels[i];
            for j in 0..self.dimension {
                centroids[label * self.dimension + j] += data_point[j];
            }
        }

        centroids.iter_mut().enumerate().for_each(|x| {
            let idx = x.0 / self.dimension;
            if cluster_sizes[idx] > 0 {
                *x.1 /= cluster_sizes[idx] as f32;
            }
        });

        // Add size penalty term
        let mut penalties = vec![0.0; self.num_cluters];
        penalties
            .iter_mut()
            .enumerate()
            .for_each(|x| *x.1 = self.tolerance * cluster_sizes[x.0] as f32);

        for _iteration in 0..self.max_iter {
            let old_labels = cluster_labels.clone();

            // Reassign points using modified distance (Equation 8)

            cluster_labels = data_points
                .par_iter()
                .map(|data_point| {
                    let dp = *data_point;
                    // Calculate distance to each centroid
                    let mut min_cost = f32::MAX;
                    let mut label = 0;
                    for centroid_id in 0..self.num_cluters {
                        let centroid = centroids
                            [centroid_id * self.dimension..(centroid_id + 1) * self.dimension]
                            .as_ref();
                        let distance = T::calculate_squared(dp, centroid) + penalties[centroid_id];

                        if distance < min_cost {
                            min_cost = distance;
                            label = centroid_id;
                        }
                    }
                    label
                })
                .collect::<Vec<usize>>();

            // Reinitialize cluster sizes
            cluster_sizes.iter_mut().for_each(|x| *x = 0);
            for i in 0..num_data_points {
                cluster_sizes[cluster_labels[i]] += 1;
            }

            // Flattened centroids
            centroids.iter_mut().for_each(|x| *x = 0.0);
            for i in 0..num_data_points {
                let data_point = &data_points[i];
                let label = cluster_labels[i];
                for j in 0..self.dimension {
                    centroids[label * self.dimension + j] += data_point[j];
                }
            }

            centroids.iter_mut().enumerate().for_each(|x| {
                let idx = x.0 / self.dimension;
                if cluster_sizes[idx] > 0 {
                    *x.1 /= cluster_sizes[idx] as f32;
                }
            });

            // Compute largest cluster
            let largest_cluster = cluster_sizes.iter().max().unwrap();
            let largest_cluster_id = cluster_sizes
                .iter()
                .position(|x| x == largest_cluster)
                .unwrap();
            let chosen_point = cluster_labels
                .iter()
                .position(|x| *x == largest_cluster_id)
                .unwrap();

            // Handle empty clusters
            for i in 0..self.num_cluters {
                if cluster_sizes[i] == 0 {
                    // Set the centroid of this cluster to the point
                    for j in 0..self.dimension {
                        centroids[i * self.dimension + j] = data_points[chosen_point][j];
                    }
                    cluster_sizes[i] = 1;
                }
            }

            // Add size penalty term
            let mut penalties = vec![0.0; self.num_cluters];
            penalties
                .iter_mut()
                .enumerate()
                .for_each(|x| *x.1 = self.tolerance * cluster_sizes[x.0] as f32);

            // Check convergence
            if cluster_labels == old_labels {
                debug!("Converged at iteration {}", _iteration);
                break;
            }
        }

        Ok(KMeansResult {
            centroids: centroids,
            assignments: cluster_labels,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmeans_lloyd() {
        let data = vec![
            vec![0.0, 0.0],
            vec![40.0, 40.0],
            vec![90.0, 90.0],
            vec![1.0, 1.0],
            vec![41.0, 41.0],
            vec![91.0, 91.0],
            vec![2.0, 2.0],
            vec![42.0, 42.0],
            vec![92.0, 92.0],
        ];

        let flattened_data = data
            .iter()
            .map(|x| x.as_slice())
            .flatten()
            .cloned()
            .collect();

        let kmeans = KMeansBuilder::new_with_cluster_init_values(
            3,
            100,
            1e-4,
            2,
            KMeansVariant::Lloyd,
            vec![0, 0, 0, 1, 1, 1, 2, 2, 2],
        );
        let result = kmeans
            .fit(flattened_data)
            .expect("KMeans run should succeed");

        assert_eq!(kmeans.num_cluters, 3);
        assert_eq!(kmeans.max_iter, 100);
        assert_eq!(kmeans.tolerance, 1e-4);
        assert_eq!(kmeans.dimension, 2);
        assert_eq!(kmeans.variant, KMeansVariant::Lloyd);

        assert_eq!(result.centroids.len(), 3 * 2);
        assert_eq!(result.assignments[0], result.assignments[3]);
        assert_eq!(result.assignments[0], result.assignments[6]);
        assert_eq!(result.assignments[1], result.assignments[4]);
        assert_eq!(result.assignments[1], result.assignments[7]);
        assert_eq!(result.assignments[2], result.assignments[5]);
        assert_eq!(result.assignments[2], result.assignments[8]);
    }

    #[test]
    fn test_kmeans_lloyd_really_large_penalty() {
        // This test tests the fact that, point (5.0, 5.0) is assigned to cluster 2 even though
        // it is supposed to be assigned to cluster 1. The penalty for unbalancing a cluster is
        // extremely large, which forces the point to be reassigned to a different cluster.
        let data = vec![
            vec![0.0, 0.0],
            vec![40.0, 40.0],
            vec![90.0, 90.0],
            vec![1.0, 1.0],
            vec![41.0, 41.0],
            vec![91.0, 91.0],
            vec![2.0, 2.0],
            vec![5.0, 5.0],
            vec![92.0, 92.0],
        ];

        let flattened_data = data
            .iter()
            .map(|x| x.as_slice())
            .flatten()
            .cloned()
            .collect();
        let kmeans = KMeansBuilder::new_with_cluster_init_values(
            3,
            100,
            10000.0,
            2,
            KMeansVariant::Lloyd,
            vec![0, 0, 0, 0, 0, 0, 2, 2, 2],
        );
        let result = kmeans
            .fit(flattened_data)
            .expect("KMeans run should succeed");

        assert_eq!(result.centroids.len(), 3 * 2);
        assert_eq!(result.assignments[0], result.assignments[3]);
        assert_eq!(result.assignments[0], result.assignments[6]);
        assert_eq!(result.assignments[1], result.assignments[4]);
        assert_eq!(result.assignments[1], result.assignments[7]);
        assert_eq!(result.assignments[2], result.assignments[5]);
        assert_eq!(result.assignments[2], result.assignments[8]);
    }

    #[test]
    fn test_kmeans_no_distance_penalty() {
        let data = vec![
            vec![0.0, 0.0],
            vec![40.0, 40.0],
            vec![90.0, 90.0],
            vec![1.0, 1.0],
            vec![41.0, 41.0],
            vec![91.0, 91.0],
            vec![2.0, 2.0],
            vec![5.0, 5.0],
            vec![92.0, 92.0],
        ];

        let flattened_data = data
            .iter()
            .map(|x| x.as_slice())
            .flatten()
            .cloned()
            .collect();
        let kmeans = KMeansBuilder::new_with_cluster_init_values(
            3,
            100,
            0.0,
            2,
            KMeansVariant::Lloyd,
            vec![0, 0, 0, 1, 1, 1, 2, 2, 2],
        );

        let result = kmeans
            .fit(flattened_data)
            .expect("KMeans run should succeed");

        assert_eq!(result.centroids.len(), 3 * 2);
        assert_eq!(result.assignments[0], result.assignments[3]);
        assert_eq!(result.assignments[0], result.assignments[6]);
        assert_eq!(result.assignments[1], result.assignments[4]);
        // point 7 belongs to the same cluster as point 0 and 3 and 6, since there is no
        // unbalanced penalty.
        assert_eq!(result.assignments[0], result.assignments[7]);
        assert_eq!(result.assignments[2], result.assignments[5]);
        assert_eq!(result.assignments[2], result.assignments[8]);
    }
}
