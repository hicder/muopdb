use anyhow::Result;
use quantization::noq::noq::NoQuantizer;
use utils::distance::l2::L2DistanceCalculator;

use super::index::Spann;
use crate::hnsw::reader::HnswReader;
use crate::ivf::reader::IvfReader;

pub struct SpannReader {
    base_directory: String,
}

impl SpannReader {
    pub fn new(base_directory: String) -> Self {
        Self { base_directory }
    }

    pub fn read(&self) -> Result<Spann> {
        let posting_list_path = format!("{}/ivf", self.base_directory);
        let centroid_path = format!("{}/centroids", self.base_directory);

        let centroids = HnswReader::new(centroid_path).read::<NoQuantizer>()?;
        let posting_lists =
            IvfReader::new(posting_list_path).read::<NoQuantizer, L2DistanceCalculator>()?;

        Ok(Spann::new(centroids, posting_lists))
    }
}

#[cfg(test)]
mod tests {

    use tempdir::TempDir;
    use utils::test_utils::generate_random_vector;

    use super::*;
    use crate::spann::builder::{SpannBuilder, SpannBuilderConfig};
    use crate::spann::writer::SpannWriter;

    #[test]
    fn test_read() {
        let temp_dir = TempDir::new("test_read").unwrap();
        let base_directory = temp_dir.path().to_str().unwrap().to_string();
        let num_clusters = 10;
        let num_vectors = 1000;
        let num_features = 4;
        let file_size = 4096;
        let balance_factor = 0.0;
        let max_posting_list_size = usize::MAX;
        let mut builder = SpannBuilder::new(SpannBuilderConfig {
            max_neighbors: 10,
            max_layers: 2,
            ef_construction: 100,
            vector_storage_memory_size: 1024,
            vector_storage_file_size: file_size,
            num_features,
            max_iteration: 1000,
            batch_size: 4,
            num_clusters,
            num_data_points: num_vectors,
            max_clusters_per_vector: 1,
            distance_threshold: 0.1,
            base_directory: base_directory.clone(),
            memory_size: 1024,
            file_size,
            tolerance: balance_factor,
            max_posting_list_size,
            reindex: false,
        })
        .unwrap();

        // Generate 1000 vectors of f32, dimension 4
        for i in 0..num_vectors {
            builder
                .add(i as u64, &generate_random_vector(num_features))
                .unwrap();
        }
        builder.build().unwrap();
        let spann_writer = SpannWriter::new(base_directory.clone());
        spann_writer.write(&mut builder).unwrap();

        let spann_reader = SpannReader::new(base_directory.clone());
        let spann = spann_reader.read().unwrap();

        let centroids = spann.get_centroids();
        let posting_lists = spann.get_posting_lists();
        assert_eq!(
            posting_lists.num_clusters,
            centroids.vector_storage.num_vectors
        );
    }
}
