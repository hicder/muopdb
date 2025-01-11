use std::cmp::Ordering;

use compression::noc::noc::PlainDecoder;
use log::debug;
use quantization::noq::noq::NoQuantizer;
use quantization::quantization::Quantizer;
use utils::distance::l2::L2DistanceCalculator;

use crate::hnsw::index::Hnsw;
use crate::index::Searchable;
use crate::ivf::index::Ivf;

pub struct Spann<Q: Quantizer> {
    centroids: Hnsw<NoQuantizer<L2DistanceCalculator>>,
    posting_lists: Ivf<Q, L2DistanceCalculator, PlainDecoder>,
}

impl<Q: Quantizer> Spann<Q> {
    pub fn new(
        centroids: Hnsw<NoQuantizer<L2DistanceCalculator>>,
        posting_lists: Ivf<Q, L2DistanceCalculator, PlainDecoder>,
    ) -> Self {
        Self {
            centroids,
            posting_lists,
        }
    }

    pub fn get_centroids(&self) -> &Hnsw<NoQuantizer<L2DistanceCalculator>> {
        &self.centroids
    }

    pub fn get_posting_lists(&self) -> &Ivf<Q, L2DistanceCalculator, PlainDecoder> {
        &self.posting_lists
    }
}

impl<Q: Quantizer> Searchable for Spann<Q> {
    fn search(
        &self,
        query: &[f32],
        k: usize,
        ef_construction: u32,
        context: &mut crate::utils::SearchContext,
    ) -> Option<Vec<crate::utils::IdWithScore>> {
        // TODO(hicder): Fully implement SPANN, which includes adjusting number of centroids
        match self.centroids.search(query, k, ef_construction, context) {
            Some(nearest_centroids) => {
                if nearest_centroids.is_empty() {
                    return None;
                }

                // Get the nearest centroid, and only search those that are within 10% of the distance of the nearest centroid
                let nearest_distance = nearest_centroids
                    .iter()
                    .map(|pad| pad.score)
                    .min_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Greater))
                    .expect("nearest_distance should not be None");

                let nearest_centroid_ids: Vec<usize> = nearest_centroids
                    .iter()
                    .filter(|centroid_and_distance| {
                        centroid_and_distance.score - nearest_distance <= nearest_distance * 0.1
                    })
                    .map(|x| x.id as usize)
                    .collect();

                debug!(
                    "Number of nearest centroids: {}",
                    nearest_centroid_ids.len()
                );

                let results = self.posting_lists.search_with_centroids_and_remap(
                    query,
                    nearest_centroid_ids,
                    k,
                    context,
                );
                Some(results)
            }
            None => None,
        }
    }
}

// TODO(hicder): Add tests
