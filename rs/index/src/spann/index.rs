use quantization::noq::noq::NoQuantizer;
use utils::distance::l2::L2DistanceCalculator;

use crate::hnsw::index::Hnsw;
use crate::index::Searchable;
use crate::ivf::index::Ivf;

pub struct Spann {
    centroids: Hnsw<NoQuantizer>,
    posting_lists: Ivf<NoQuantizer, L2DistanceCalculator>,
}

impl Spann {
    pub fn new(
        centroids: Hnsw<NoQuantizer>,
        posting_lists: Ivf<NoQuantizer, L2DistanceCalculator>,
    ) -> Self {
        Self {
            centroids,
            posting_lists,
        }
    }

    pub fn get_centroids(&self) -> &Hnsw<NoQuantizer> {
        &self.centroids
    }

    pub fn get_posting_lists(&self) -> &Ivf<NoQuantizer, L2DistanceCalculator> {
        &self.posting_lists
    }
}

impl Searchable for Spann {
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
                let nearest_centroid_ids =
                    nearest_centroids.iter().map(|x| x.id as usize).collect();
                if nearest_centroids.is_empty() {
                    return None;
                }
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
