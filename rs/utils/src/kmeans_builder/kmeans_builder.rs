use std::cmp::min;
use std::marker::PhantomData;
use std::simd::{LaneCount, Simd, SupportedLaneCount};

use anyhow::{anyhow, Ok, Result};
use kmeans::KMeansConfig;
use log::debug;
use rand::seq::SliceRandom;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use rayon::slice::ParallelSlice;

use crate::distance::lane_conforming::LaneConformingDistanceCalculator;
use crate::{CalculateSquared, DistanceCalculator};

#[derive(PartialEq, Debug)]
pub enum KMeansVariant {
    Lloyd,
}

pub struct KMeansBuilder<D: DistanceCalculator + CalculateSquared + Send + Sync> {
    pub num_clusters: usize,
    pub max_iter: usize,

    // Factor which determine how much penalty large cluster has over small cluster.
    pub tolerance: f32,

    // data shape
    pub dimension: usize,

    // Variant for this algorithm. Currently only Lloyd is supported.
    pub variant: KMeansVariant,

    pub cluster_init_values: Option<Vec<usize>>,

    _marker: PhantomData<D>,
}

pub struct KMeansResult {
    // Flattened centroids
    pub centroids: Vec<f32>,
    pub assignments: Vec<usize>,
    pub error: f32,
}

// TODO(hicder): Add support for different variants of k-means.
// TODO(hicder): Add support for different distance metrics.
impl<D: DistanceCalculator + CalculateSquared + Send + Sync> KMeansBuilder<D> {
    pub fn new(
        num_cluters: usize,
        max_iter: usize,
        tolerance: f32,
        dimension: usize,
        variant: KMeansVariant,
    ) -> Self {
        Self {
            num_clusters: num_cluters,
            max_iter,
            tolerance,
            dimension,
            variant,
            cluster_init_values: None,
            _marker: PhantomData,
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
            num_clusters: num_cluters,
            max_iter,
            tolerance,
            dimension,
            variant,
            cluster_init_values: Some(cluster_init_values),
            _marker: PhantomData,
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
            self.num_clusters,
            self.max_iter,
            kmeans::KMeans::init_random_sample,
            &conf,
        );
        let kmeans_result = KMeansResult {
            // intial_centroids: vec![],
            centroids: result.centroids,
            assignments: result.assignments,
            error: result.distsum,
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
                        .run_lloyd::<LaneConformingDistanceCalculator<16, D>, 16>(flattened_data);
                } else if self.dimension % 8 == 0 {
                    return self
                        .run_lloyd::<LaneConformingDistanceCalculator<8, D>, 8>(flattened_data);
                } else if self.dimension % 4 == 0 {
                    return self
                        .run_lloyd::<LaneConformingDistanceCalculator<4, D>, 4>(flattened_data);
                } else {
                    return self.run_lloyd::<D, 1>(flattened_data);
                }
            }
        }
    }

    fn init_random_points(&self, points: &Vec<&[f32]>, num_clusters: usize) -> Result<Vec<f32>> {
        match &self.cluster_init_values {
            Some(cluster_init_values) if cluster_init_values.len() == num_clusters => {
                return Ok(cluster_init_values
                    .iter()
                    .map(|point_id| points[*point_id])
                    .flatten()
                    .cloned()
                    .collect());
            }
            _ => {
                let mut rng = rand::thread_rng();
                let mut centroids = vec![];
                points
                    .choose_multiple(&mut rng, num_clusters)
                    .for_each(|point| {
                        centroids.extend_from_slice(*point);
                    });
                return Ok(centroids);
            }
        }
    }

    fn run_lloyd<T: CalculateSquared + Send + Sync, const SIMD_WIDTH: usize>(
        &self,
        flattened_data_points: Vec<f32>,
    ) -> Result<KMeansResult>
    where
        LaneCount<SIMD_WIDTH>: SupportedLaneCount,
    {
        let data_points = flattened_data_points
            .par_chunks_exact(self.dimension)
            .map(|x| x)
            .collect::<Vec<&[f32]>>();

        let num_data_points = data_points.len();
        let num_clusters = min(self.num_clusters, num_data_points);

        // Choose random few points as initial centroids
        let mut centroids = self.init_random_points(&data_points, num_clusters)?;
        let mut cluster_sizes = vec![0; num_clusters];

        // Add size penalty term
        let mut penalties = vec![0.0; num_clusters];
        if self.tolerance > 0.0 {
            penalties
                .iter_mut()
                .enumerate()
                .for_each(|x| *x.1 = self.tolerance * cluster_sizes[x.0] as f32);
        }

        let mut cluster_labels = vec![0; data_points.len()];

        let mut last_dist = f32::MAX;
        let mut iteration = 0;
        loop {
            let last_labels = cluster_labels.clone();

            // Reassign points using modified distance (Equation 8)
            let mut cluster_labels_with_min_cost = data_points
                .par_iter()
                .map(|data_point| {
                    let dp = *data_point;
                    // Calculate distance to each centroid
                    let res = centroids
                        .chunks_exact(self.dimension)
                        .enumerate()
                        .map(|(centroid_id, centroid)| {
                            let distance =
                                T::calculate_squared(dp, centroid) + penalties[centroid_id];
                            (centroid_id, distance)
                        })
                        .fold((0, f32::MAX), |(min_label, min_cost), (label, distance)| {
                            if distance < min_cost {
                                (label, distance)
                            } else {
                                (min_label, min_cost)
                            }
                        });
                    res
                })
                .collect::<Vec<(usize, f32)>>();

            let mut total_dist = 0.0;
            rayon::scope(|s| {
                s.spawn(|_| {
                    total_dist = cluster_labels_with_min_cost
                        .iter()
                        .map(|(_, distance)| (*distance).sqrt())
                        .sum::<f32>();
                });
                s.spawn(|_| {
                    centroids.iter_mut().for_each(|x| *x = 0.0);
                    data_points
                        .iter()
                        .zip(cluster_labels_with_min_cost.iter())
                        .for_each(|(data_point, (label, _))| {
                            let dp = *data_point;
                            let centroid_slice = &mut centroids
                                [*label * self.dimension..(label + 1) * self.dimension];
                            centroid_slice
                                .chunks_exact_mut(SIMD_WIDTH)
                                .zip(
                                    dp.chunks_exact(SIMD_WIDTH)
                                        .map(|v| Simd::<f32, SIMD_WIDTH>::from_slice(v)),
                                )
                                .for_each(|(c, s)| {
                                    let c_simd = Simd::<f32, SIMD_WIDTH>::from_slice(c);
                                    let result = c_simd + s;
                                    c.copy_from_slice(result.as_array());
                                });
                        });
                });
                s.spawn(|_| {
                    cluster_sizes.iter_mut().for_each(|x| *x = 0);
                    for i in 0..num_data_points {
                        cluster_sizes[cluster_labels_with_min_cost[i].0] += 1;
                    }
                });
            });

            let mut contains_empty_cluster = false;
            // Reinitialize cluster sizes
            centroids.iter_mut().enumerate().for_each(|x| {
                let idx = x.0 / self.dimension;
                if cluster_sizes[idx] > 0 {
                    *x.1 /= cluster_sizes[idx] as f32;
                } else {
                    contains_empty_cluster = true;
                }
            });

            // Check if there is any empty cluster
            if contains_empty_cluster {
                // Find the point that is in a cluster with more than 1 points, that is farthest away from its cluster
                for cluster_id in 0..cluster_sizes.len() {
                    if cluster_sizes[cluster_id] == 0 {
                        // debug!("Cluster {} with 0 points found", cluster_id);
                        let mut max_distance = 0.0;
                        let mut chosen_point_id = 0;
                        let mut chosen_cluster_id = 0;
                        for i in 0..cluster_labels_with_min_cost.len() {
                            let checking_cluster_id = cluster_labels_with_min_cost[i].0;
                            if cluster_sizes[checking_cluster_id] > 1 {
                                let point = data_points[i];
                                let cluster = centroids
                                    .chunks_exact(self.dimension)
                                    .nth(cluster_id)
                                    .unwrap();
                                let distance = T::calculate_squared(point, cluster);
                                if distance > max_distance {
                                    max_distance = distance;
                                    chosen_point_id = i;
                                    chosen_cluster_id = checking_cluster_id;
                                }
                            }
                        }

                        let old_size = cluster_sizes[chosen_cluster_id] as f32;
                        cluster_sizes[chosen_cluster_id] -= 1;

                        let chosen_point = data_points[chosen_point_id];
                        for j in 0..self.dimension {
                            let x = centroids[chosen_cluster_id * self.dimension + j];
                            centroids[chosen_cluster_id * self.dimension + j] =
                                (x * old_size - chosen_point[j]) / (old_size - 1.0);
                        }

                        // add chosen point to the new cluster
                        cluster_labels_with_min_cost[chosen_point_id].0 = cluster_id;
                        cluster_sizes[cluster_id] = 1;
                        // update centroid for this cluster
                        for j in 0..self.dimension {
                            centroids[cluster_id * self.dimension + j] = chosen_point[j];
                        }
                    }
                }
            }

            // Add size penalty term
            if self.tolerance > 0.0 {
                penalties = vec![0.0; num_clusters];
                penalties
                    .iter_mut()
                    .enumerate()
                    .for_each(|x| *x.1 = self.tolerance * cluster_sizes[x.0] as f32);

                total_dist += penalties
                    .iter()
                    .zip(cluster_sizes.iter())
                    .map(|(penalty, size)| *penalty * (*size as f32))
                    .sum::<f32>();
            }

            debug!(
                "Iteration: {}, Error {} -> {}, improvement: {}",
                iteration,
                last_dist,
                total_dist,
                last_dist - total_dist
            );
            // TODO(hicder): Make 0.0005 a parameter
            cluster_labels = cluster_labels_with_min_cost
                .iter()
                .map(|(label, _)| *label)
                .collect();
            if cluster_labels == last_labels || iteration >= self.max_iter {
                debug!(
                    "Converged at iteration {}, improvement: {}",
                    iteration,
                    total_dist - last_dist
                );
                break;
            }
            last_dist = total_dist;
            iteration += 1;
        }

        Ok(KMeansResult {
            centroids: centroids,
            assignments: cluster_labels,
            error: last_dist,
        })
    }
}

#[cfg(test)]
mod tests {

    use std::collections::HashSet;

    use super::*;
    use crate::distance::l2::L2DistanceCalculator;

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

        let kmeans = KMeansBuilder::<L2DistanceCalculator>::new_with_cluster_init_values(
            3,
            100,
            1e-4,
            2,
            KMeansVariant::Lloyd,
            vec![0, 1, 2],
        );
        let result = kmeans
            .fit(flattened_data)
            .expect("KMeans run should succeed");

        assert_eq!(kmeans.num_clusters, 3);
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
        let kmeans = KMeansBuilder::<L2DistanceCalculator>::new_with_cluster_init_values(
            3,
            100,
            0.0,
            2,
            KMeansVariant::Lloyd,
            vec![0, 1, 2],
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

    #[test]
    fn test_kmeans_with_empty_cluster() {
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
        let kmeans = KMeansBuilder::<L2DistanceCalculator>::new_with_cluster_init_values(
            10,
            100,
            0.0,
            2,
            KMeansVariant::Lloyd,
            vec![0, 1, 2],
        );

        let result = kmeans
            .fit(flattened_data)
            .expect("KMeans run should succeed");

        assert_eq!(result.centroids.len(), 9 * 2);
        let asigned_clusters: HashSet<usize> =
            result.assignments.iter().map(|x| *x as usize).collect();
        let expected_clusters: HashSet<usize> = HashSet::from([0, 1, 2, 3, 4, 5, 6, 7, 8]);

        assert_eq!(asigned_clusters, expected_clusters);
    }
}
