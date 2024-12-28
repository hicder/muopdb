use std::sync::Arc;

use anyhow::{Ok, Result};
use utils::io::get_latest_version;

use super::{Collection, TableOfContent};
use crate::collection::BoxedSegmentSearchable;
use crate::segment::immutable_segment::ImmutableSegment;
use crate::spann::builder::SpannBuilderConfig;
use crate::spann::reader::SpannReader;

pub struct Reader {
    path: String,
}

impl Reader {
    pub fn new(path: String) -> Self {
        Self { path }
    }

    pub fn read(&self) -> Result<Arc<Collection>> {
        // Read the SpannBuilderConfig
        let spann_builder_config_path = format!("{}/collection_config.json", self.path);
        let spann_builder_config: SpannBuilderConfig =
            serde_json::from_reader(std::fs::File::open(spann_builder_config_path)?)?;

        // Get the latest TOC
        let latest_version = get_latest_version(&self.path)?;
        let toc_path = format!("{}/version_{}", self.path, latest_version);
        let toc: TableOfContent = serde_json::from_reader(std::fs::File::open(toc_path)?)?;

        // let collection = Arc::new(Collection::new(self.path.clone()));
        let mut segments: Vec<Arc<BoxedSegmentSearchable>> = vec![];
        for name in &toc.toc {
            let spann_path = format!("{}/{}", self.path, name);
            let spann_reader = SpannReader::new(spann_path);
            let index = spann_reader.read()?;
            segments.push(Arc::new(Box::new(ImmutableSegment::new(index))));
        }

        let collection = Arc::new(Collection::init_from(
            self.path.clone(),
            latest_version,
            toc,
            segments,
            spann_builder_config,
        )?);
        Ok(collection)
    }
}

// TODO(hicder): Add tests once I write builder and writer for SPANN.
#[cfg(test)]
mod tests {
    use anyhow::Result;
    use tempdir::TempDir;
    use utils::test_utils::generate_random_vector;

    use super::*;
    use crate::spann::builder::{SpannBuilder, SpannBuilderConfig};
    use crate::spann::writer::SpannWriter;

    fn collection_config() -> SpannBuilderConfig {
        SpannBuilderConfig {
            max_neighbors: 10,
            max_layers: 2,
            ef_construction: 100,
            vector_storage_memory_size: 1024,
            vector_storage_file_size: 1024,
            num_features: 4,
            max_iteration: 1000,
            batch_size: 4,
            num_clusters: 10,
            num_data_points: 1000,
            max_clusters_per_vector: 1,
            distance_threshold: 0.1,
            base_directory: "./".to_string(),
            memory_size: 1024,
            file_size: 1024,
            tolerance: 0.1,
            max_posting_list_size: usize::MAX,
            reindex: true,
        }
    }

    fn create_segment(base_directory: String) -> Result<()> {
        let num_vectors = 1000;
        let num_features = 4;
        let mut collection_config = collection_config();
        collection_config.base_directory = base_directory.clone();
        let mut builder = SpannBuilder::new(collection_config).unwrap();

        // Generate 1000 vectors of f32, dimension 4
        for i in 0..num_vectors {
            builder
                .add(i as u64, &generate_random_vector(num_features))
                .unwrap();
        }
        builder.build().unwrap();
        let spann_writer = SpannWriter::new(base_directory.clone());
        spann_writer.write(&mut builder)?;

        Ok(())
    }

    #[test]
    fn test_reader() {
        let temp_dir = TempDir::new("test_reader").unwrap();
        let base_directory: String = temp_dir.path().to_str().unwrap().to_string();

        // Write the collection config
        let collection_config_path = format!("{}/collection_config.json", base_directory);
        let collection_config = collection_config();
        serde_json::to_writer(
            std::fs::File::create(collection_config_path).unwrap(),
            &collection_config,
        )
        .unwrap();

        // Create "segment1"
        let segment1_path = format!("{}/segment1", base_directory);
        std::fs::create_dir_all(&segment1_path).unwrap();
        create_segment(segment1_path).unwrap();
        // Create "segment2"
        let segment2_path = format!("{}/segment2", base_directory);
        std::fs::create_dir_all(&segment2_path).unwrap();
        create_segment(segment2_path).unwrap();

        // Create a TOC version 0
        let toc_path = format!("{}/version_0", base_directory);
        let toc = TableOfContent::new(vec!["segment1".to_string()]);
        serde_json::to_writer(std::fs::File::create(toc_path).unwrap(), &toc).unwrap();

        // Create a TOC version 1
        let toc_path = format!("{}/version_1", base_directory);
        let toc = TableOfContent::new(vec!["segment1".to_string(), "segment2".to_string()]);
        serde_json::to_writer(std::fs::File::create(toc_path).unwrap(), &toc).unwrap();

        let reader = Reader::new(base_directory.clone());
        let collection = reader.read().unwrap();

        // Check current version
        assert_eq!(collection.current_version(), 1);

        // Get current snapshot
        let snapshot = collection.get_snapshot().unwrap();
        assert_eq!(snapshot.segments.len(), 2);
    }
}