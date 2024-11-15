use std::collections::{BinaryHeap, HashMap};

use utils::l2::L2DistanceCalculator;
use utils::DistanceCalculator;

use crate::index::Index;
use crate::utils::{IdWithScore, SearchContext};

pub struct Ivf {
    // The whole dataset. TODO(tyb0807): reduce the memory footprint.
    pub dataset: Vec<Vec<f32>>,
    // Number of clusters.
    pub num_clusters: usize,
    // Each cluster is represented by a centroid vector. This is all the centroids in our IVF.
    pub centroids: Vec<Vec<f32>>,
    // Inverted index mapping each cluster to the vectors it contains.
    //   key: centroid index in `centroids`
    //   value: vector index in `dataset`
    pub inverted_lists: HashMap<usize, Vec<usize>>,
    // Number of probed centroids.
    pub num_probes: usize,
}

impl Ivf {
    pub fn new(
        dataset: Vec<Vec<f32>>,
        centroids: Vec<Vec<f32>>,
        inverted_lists: HashMap<usize, Vec<usize>>,
        num_clusters: usize,
        num_probes: usize,
    ) -> Self {
        Self {
            dataset,
            num_clusters,
            centroids,
            inverted_lists,
            num_probes,
        }
    }

    fn find_nearest_centroids(
        vector: &Vec<f32>,
        centroids: &Vec<Vec<f32>>,
        num_probes: usize,
    ) -> Vec<usize> {
        let mut calculator = L2DistanceCalculator::new();
        let mut distances: Vec<(usize, f32)> = centroids
            .iter()
            .enumerate()
            .map(|(idx, centroid)| {
                let dist = calculator.calculate(&vector, &centroid);
                (idx, dist)
            })
            .collect();
        distances.select_nth_unstable_by(num_probes - 1, |a, b| a.1.partial_cmp(&b.1).unwrap());
        distances.into_iter().map(|(idx, _)| idx).collect()
    }

    pub fn build_inverted_lists(
        dataset: &Vec<Vec<f32>>,
        centroids: &Vec<Vec<f32>>,
    ) -> HashMap<usize, Vec<usize>> {
        let mut inverted_lists = HashMap::new();
        // Assign vectors to nearest centroid
        for (i, vector) in dataset.iter().enumerate() {
            let nearest_centroid = Self::find_nearest_centroids(&vector, &centroids, 1);
            inverted_lists
                .entry(nearest_centroid[0])
                .or_insert_with(Vec::new)
                .push(i);
        }
        inverted_lists
    }
}

impl Index for Ivf {
    fn search(
        &self,
        query: &[f32],
        k: usize,
        _ef_construction: u32,
        _context: &mut SearchContext,
    ) -> Option<Vec<IdWithScore>> {
        // TODO(tyb0807): maybe do something with `context`.
        let mut heap = BinaryHeap::with_capacity(k);

        // Find the nearest centroids to the query.
        let nearest_centroids =
            Self::find_nearest_centroids(&query.to_vec(), &self.centroids, self.num_probes);

        // Search in the inverted lists of the nearest centroids.
        for &centroid in &nearest_centroids {
            if let Some(list) = self.inverted_lists.get(&centroid) {
                for &idx in list {
                    let distance = L2DistanceCalculator::new().calculate(query, &self.dataset[idx]);
                    let id_with_score = IdWithScore {
                        score: distance,
                        id: idx as u64,
                    };
                    if heap.len() < k {
                        heap.push(id_with_score);
                    } else if let Some(max) = heap.peek() {
                        if id_with_score < *max {
                            heap.pop();
                            heap.push(id_with_score);
                        }
                    }
                }
            }
        }

        // Convert heap to a sorted vector in ascending order.
        let mut results: Vec<IdWithScore> = heap.into_vec();
        results.sort();
        Some(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ivf_new() {
        let dataset = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let centroids = vec![vec![1.5, 2.5, 3.5], vec![5.5, 6.5, 7.5]];
        let inverted_lists = Ivf::build_inverted_lists(&dataset, &centroids);
        let num_clusters = 2;
        let num_probes = 1;

        let ivf = Ivf::new(
            dataset.clone(),
            centroids.clone(),
            inverted_lists.clone(),
            num_clusters,
            num_probes,
        );

        assert_eq!(ivf.dataset, dataset);
        assert_eq!(ivf.num_clusters, num_clusters);
        assert_eq!(ivf.centroids, centroids);
        assert_eq!(ivf.inverted_lists, inverted_lists);
        assert_eq!(ivf.num_probes, num_probes);
        assert_eq!(ivf.inverted_lists.len(), 2);
        assert!(ivf.inverted_lists.get(&0).unwrap().contains(&0));
        assert!(ivf.inverted_lists.get(&1).unwrap().contains(&2));
    }

    #[test]
    fn test_find_nearest_centroids() {
        let vector = vec![3.0, 4.0, 5.0];
        let centroids = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let num_probes = 2;

        let nearest = Ivf::find_nearest_centroids(&vector, &centroids, num_probes);

        assert_eq!(nearest[0], 1);
        assert_eq!(nearest[1], 0);
    }

    #[test]
    fn test_search() {
        let dataset = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
            vec![2.0, 3.0, 4.0],
        ];
        let centroids = vec![vec![1.5, 2.5, 3.5], vec![5.5, 6.5, 7.5]];
        let inverted_lists = Ivf::build_inverted_lists(&dataset, &centroids);
        let num_clusters = 2;
        let num_probes = 2;

        let ivf = Ivf::new(dataset, centroids, inverted_lists, num_clusters, num_probes);

        let query = vec![2.0, 3.0, 4.0];
        let k = 2;
        let ef_construction = 10;
        let mut context = SearchContext::new(false);

        let results = ivf
            .search(&query, k, ef_construction, &mut context)
            .unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, 3); // Closest to [2.0, 3.0, 4.0]
        assert_eq!(results[1].id, 0); // Second closest to [2.0, 3.0, 4.0]
        assert!(results[0].score < results[1].score);
    }

    #[test]
    fn test_search_with_empty_result() {
        let dataset = vec![vec![100.0, 200.0, 300.0]];
        let centroids = vec![vec![100.0, 200.0, 300.0]];
        let inverted_lists = Ivf::build_inverted_lists(&dataset, &centroids);
        let num_clusters = 1;
        let num_probes = 1;

        let ivf = Ivf::new(dataset, centroids, inverted_lists, num_clusters, num_probes);

        let query = vec![1.0, 2.0, 3.0];
        let k = 5; // More than available results
        let ef_construction = 10;
        let mut context = SearchContext::new(false);

        let results = ivf
            .search(&query, k, ef_construction, &mut context)
            .unwrap();

        assert_eq!(results.len(), 1); // Only one result available
        assert_eq!(results[0].id, 0);
    }
}
