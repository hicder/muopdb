use std::vec;

use anyhow::{anyhow, Result};

use crate::l2::L2DistanceCalculator;
use crate::DistanceCalculator;

#[derive(PartialEq, Debug)]
pub enum KMeansVariant {
    Lloyd,
}

pub struct KMeans {
    pub num_cluters: usize,
    pub max_iter: usize,

    // Factor which determine how much penalty large cluster has over small cluster.
    pub tolerance: f32,

    // data shape
    pub dimension: usize,

    // Variant for this algorithm. Currently only Lloyd is supported.
    pub variant: KMeansVariant,
}

pub struct KMeansResult {
    // Flattened centroids
    pub centroids: Vec<f32>,
    pub assignments: Vec<usize>,
}

// TODO(hicder): Add support for different variants of k-means.
// TODO(hicder): Add support for different distance metrics.
// TODO(hicder): Use SIMD for computation.
impl KMeans {
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
        }
    }

    pub fn fit(&self, data: Vec<&[f32]>) -> Result<KMeansResult> {
        // Validate dimension
        for data_point in data.iter() {
            if data_point.len() != self.dimension {
                return Err(anyhow!(
                    "Dimension of data point {} is not equal to dimension of KMeans object {}",
                    data_point.len(),
                    self.dimension
                ));
            }
        }

        match self.variant {
            KMeansVariant::Lloyd => self.run_lloyd(data),
        }
    }

    fn run_lloyd(&self, data_points: Vec<&[f32]>) -> Result<KMeansResult> {
        let num_data_points = data_points.len();
        let mut cluster_labels = vec![0; num_data_points];

        // Random initialization of cluster labels
        for i in 0..num_data_points {
            cluster_labels[i] = rand::random::<usize>() % self.num_cluters;
        }

        let mut final_centroids = vec![0.0; self.num_cluters * self.dimension];

        for _iteration in 0..self.max_iter {
            let old_labels = cluster_labels.clone();

            // Calculate current cluster sizes
            let mut cluster_sizes = vec![0; self.num_cluters];
            for i in 0..num_data_points {
                cluster_sizes[old_labels[i]] += 1;
            }

            // Flattened centroids
            let mut centroids = vec![0.0; self.num_cluters * self.dimension];
            for i in 0..num_data_points {
                let data_point = &data_points[i];
                let label = old_labels[i];
                for j in 0..self.dimension {
                    centroids[label * self.dimension + j] += data_point[j];
                }
            }
            centroids
                .iter_mut()
                .enumerate()
                .for_each(|x| *x.1 /= cluster_sizes[x.0 / self.dimension] as f32);

            final_centroids = centroids.clone();

            // Add size penalty term
            let mut penalties = vec![0.0; self.num_cluters];
            penalties
                .iter_mut()
                .enumerate()
                .for_each(|x| *x.1 = self.tolerance * cluster_sizes[x.0] as f32);

            // Reassign points using modified distance (Equation 8)
            for i in 0..num_data_points {
                let mut distances = vec![0.0; self.num_cluters];
                let data_point = &data_points[i];

                // Calculate distance to each centroid
                for centroid_id in 0..self.num_cluters {
                    let centroid = centroids
                        [centroid_id * self.dimension..(centroid_id + 1) * self.dimension]
                        .as_ref();
                    let mut distance_calculator = L2DistanceCalculator::new();
                    let distance = distance_calculator.calculate(data_point, centroid);
                    distances[centroid_id] = distance;
                }

                // Assign each point to cluster with minimum cost (which includes size penalty)
                let mut min_cost = f32::MAX;
                for j in 0..self.num_cluters {
                    let cost = distances[j] + penalties[j];
                    if cost < min_cost {
                        min_cost = cost;
                        cluster_labels[i] = j;
                    }
                }
            }

            // Check convergence
            if cluster_labels == old_labels {
                break;
            }
        }

        Ok(KMeansResult {
            centroids: final_centroids,
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

        let kmeans = KMeans::new(3, 100, 1e-4, 2, KMeansVariant::Lloyd);
        let data_ref = data.iter().map(|x| x.as_slice()).collect();
        let result = kmeans.fit(data_ref).unwrap();

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



        let kmeans = KMeans::new(3, 100, 10000.0, 2, KMeansVariant::Lloyd);
        let data_ref = data.iter().map(|x| x.as_slice()).collect();
        let result = kmeans.fit(data_ref).unwrap();

        assert_eq!(result.centroids.len(), 3 * 2);
        assert_eq!(result.assignments[0], result.assignments[3]);
        assert_eq!(result.assignments[0], result.assignments[6]);
        assert_eq!(result.assignments[1], result.assignments[4]);
        assert_eq!(result.assignments[1], result.assignments[7]);
        assert_eq!(result.assignments[2], result.assignments[5]);
        assert_eq!(result.assignments[2], result.assignments[8]);

    }
}
