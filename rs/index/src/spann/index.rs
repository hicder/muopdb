use std::cmp::Ordering;

use compression::noc::noc::PlainDecoder;
use log::debug;
use quantization::noq::noq::NoQuantizer;
use quantization::quantization::Quantizer;
use utils::distance::l2::L2DistanceCalculator;

use crate::hnsw::index::Hnsw;
use crate::index::Searchable;
use crate::ivf::index::Ivf;
use crate::utils::SearchContext;

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

    pub fn get_doc_id(&self, point_id: u32) -> Option<u128> {
        match self
            .posting_lists
            .index_storage
            .get_doc_id(point_id as usize)
        {
            Ok(doc_id) => Some(doc_id),
            Err(_) => None,
        }
    }

    pub fn get_vector(
        &self,
        point_id: u32,
        context: &mut SearchContext,
    ) -> Option<&[Q::QuantizedT]> {
        self.posting_lists
            .vector_storage
            .get(point_id as usize, context)
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

#[cfg(test)]
mod tests {
    use config::enums::{IntSeqEncodingType, QuantizerType};
    use quantization::noq::noq::NoQuantizer;
    use quantization::pq::pq::ProductQuantizer;
    use utils::distance::l2::L2DistanceCalculator;

    use super::*;
    use crate::spann::builder::{SpannBuilder, SpannBuilderConfig};
    use crate::spann::reader::SpannReader;
    use crate::spann::writer::SpannWriter;
    use crate::utils::SearchContext;

    #[test]
    fn test_spann_search() {
        let temp_dir = tempdir::TempDir::new("spann_search_test")
            .expect("Failed to create temporary directory");
        let base_dir = temp_dir
            .path()
            .to_str()
            .expect("Failed to convert temporary directory path to string")
            .to_string();

        let num_clusters = 10;
        let num_vectors = 1000;
        let num_features = 4;
        let file_size = 4096;
        let balance_factor = 0.0;
        let max_posting_list_size = usize::MAX;
        let mut builder = SpannBuilder::new(SpannBuilderConfig {
            centroids_max_neighbors: 10,
            centroids_max_layers: 2,
            centroids_ef_construction: 100,
            centroids_vector_storage_memory_size: 1024,
            centroids_vector_storage_file_size: file_size,
            num_features,
            pq_subvector_dimension: 8,
            pq_num_bits: 8,
            pq_num_training_rows: 50,
            quantizer_type: QuantizerType::NoQuantizer,
            pq_max_iteration: 1000,
            pq_batch_size: 4,
            ivf_num_clusters: num_clusters,
            ivf_num_data_points_for_clustering: num_vectors,
            ivf_max_clusters_per_vector: 1,
            ivf_distance_threshold: 0.1,
            posting_list_encoding_type: IntSeqEncodingType::PlainEncoding,
            ivf_base_directory: base_dir.clone(),
            ivf_vector_storage_memory_size: 1024,
            ivf_vector_storage_file_size: file_size,
            centroids_clustering_tolerance: balance_factor,
            ivf_max_posting_list_size: max_posting_list_size,
            reindex: false,
        })
        .unwrap();

        // Generate 1000 vectors of f32, dimension 4
        for i in 0..num_vectors {
            builder
                .add(i as u128, &vec![i as f32, i as f32, i as f32, i as f32])
                .unwrap();
        }
        assert!(builder.build().is_ok());
        let spann_writer = SpannWriter::new(base_dir.clone());
        assert!(spann_writer.write(&mut builder).is_ok());

        let spann_reader = SpannReader::new(base_dir.clone());
        let spann = spann_reader
            .read::<NoQuantizer<L2DistanceCalculator>>()
            .unwrap();

        let query = vec![2.4, 3.4, 4.4, 5.4];
        let k = 2;
        let num_probes = 2;
        let mut context = SearchContext::new(false);

        let results = spann
            .search(&query, k, num_probes, &mut context)
            .expect("IVF search should return a result");

        assert_eq!(results.len(), k);
        assert_eq!(results[0].id, 4); // Closest to [4.0, 4.0, 4.0, 4.0]
        assert_eq!(results[1].id, 3); // Next is [3.0, 3.0, 3.0, 3.0]
    }

    #[test]
    fn test_spann_search_with_pq() {
        let temp_dir = tempdir::TempDir::new("spann_search_with_pq_test")
            .expect("Failed to create temporary directory");
        let base_dir = temp_dir
            .path()
            .to_str()
            .expect("Failed to convert temporary directory path to string")
            .to_string();

        let num_clusters = 10;
        let num_vectors = 1000;
        let num_features = 4;
        let file_size = 4096;
        let balance_factor = 0.0;
        let max_posting_list_size = usize::MAX;
        let mut builder = SpannBuilder::new(SpannBuilderConfig {
            centroids_max_neighbors: 10,
            centroids_max_layers: 2,
            centroids_ef_construction: 100,
            centroids_vector_storage_memory_size: 1024,
            centroids_vector_storage_file_size: file_size,
            num_features,
            pq_subvector_dimension: 2,
            pq_num_bits: 2,
            pq_num_training_rows: 200,
            quantizer_type: QuantizerType::ProductQuantizer,
            pq_max_iteration: 1000,
            pq_batch_size: 4,
            ivf_num_clusters: num_clusters,
            ivf_num_data_points_for_clustering: num_vectors,
            ivf_max_clusters_per_vector: 1,
            ivf_distance_threshold: 0.1,
            posting_list_encoding_type: IntSeqEncodingType::PlainEncoding,
            ivf_base_directory: base_dir.clone(),
            ivf_vector_storage_memory_size: 1024,
            ivf_vector_storage_file_size: file_size,
            centroids_clustering_tolerance: balance_factor,
            ivf_max_posting_list_size: max_posting_list_size,
            reindex: false,
        })
        .unwrap();

        // Generate 1000 vectors of f32, dimension 4
        for i in 0..num_vectors {
            builder
                .add(i as u128, &vec![i as f32, i as f32, i as f32, i as f32])
                .unwrap();
        }
        assert!(builder.build().is_ok());
        let spann_writer = SpannWriter::new(base_dir.clone());
        assert!(spann_writer.write(&mut builder).is_ok());

        let spann_reader = SpannReader::new(base_dir.clone());
        let spann = spann_reader
            .read::<ProductQuantizer<L2DistanceCalculator>>()
            .unwrap();

        let query = vec![2.4, 3.4, 4.4, 5.4];
        let k = 5;
        let num_probes = 2;
        let mut context = SearchContext::new(false);

        let results = spann
            .search(&query, k, num_probes, &mut context)
            .expect("IVF search should return a result");

        assert_eq!(results.len(), k);
        // subvector_dimension = 2 so very lossy!
        assert_eq!(results[0].score, 0.0);
        assert_eq!(results[1].score, 0.0);
        assert_eq!(results[2].score, 0.0);
        assert_eq!(results[3].score, 0.0);
        assert_eq!(results[4].score, 0.0);
    }
}
