use std::sync::Arc;

use anyhow::{Ok, Result};
use config::collection::CollectionConfig;
use parking_lot::RwLock;
use quantization::quantization::Quantizer;
use utils::io::get_latest_version;

use super::{Collection, TableOfContent};
use crate::multi_spann::reader::MultiSpannReader;
use crate::segment::immutable_segment::ImmutableSegment;
use crate::segment::pending_segment::PendingSegment;
use crate::segment::{BoxedImmutableSegment, Segment};

pub struct CollectionReader {
    path: String,
}

impl CollectionReader {
    pub fn new(path: String) -> Self {
        Self { path }
    }

    pub fn read<Q: Quantizer + Clone>(&self) -> Result<Arc<Collection<Q>>> {
        // Read the SpannBuilderConfig
        let spann_builder_config_path = format!("{}/collection_config.json", self.path);
        let collection_config: CollectionConfig =
            serde_json::from_reader(std::fs::File::open(spann_builder_config_path)?)?;

        // Get the latest TOC
        let latest_version = get_latest_version(&self.path)?;
        let toc_path = format!("{}/version_{}", self.path, latest_version);
        let toc: TableOfContent = serde_json::from_reader(std::fs::File::open(toc_path)?)?;

        // Read the segments
        let mut segments: Vec<BoxedImmutableSegment<Q>> = vec![];
        for name in &toc.toc {
            // We read pending segments later
            if toc.pending.contains_key(name) {
                continue;
            }

            let spann_path = format!("{}/{}", self.path, name);
            let spann_reader = MultiSpannReader::new(spann_path);
            let index = spann_reader.read::<Q>()?;
            segments.push(BoxedImmutableSegment::FinalizedSegment(Arc::new(
                RwLock::new(ImmutableSegment::new(index, name.clone())),
            )));
        }

        // Empty all the pending segments
        let pending_segment_names = toc.pending.keys().collect::<Vec<&String>>();
        for pending_segment_name in pending_segment_names {
            let pending_segment_path = format!("{}/{}", self.path, pending_segment_name);
            std::fs::remove_dir_all(&pending_segment_path).unwrap();
            std::fs::create_dir_all(&pending_segment_path).unwrap();

            // Get the inner segments
            let inner_segment_names = toc.pending.get(pending_segment_name).unwrap();
            let mut inner_segments: Vec<BoxedImmutableSegment<Q>> = vec![];
            for inner_segment_name in inner_segment_names {
                for segment in &segments {
                    if segment.name() == *inner_segment_name {
                        inner_segments.push(segment.clone());
                        break;
                    }
                }
            }

            segments.push(BoxedImmutableSegment::PendingSegment(
                Arc::new(RwLock::new(PendingSegment::new(
                    inner_segments,
                    pending_segment_path,
                ))),
            ));
        }

        let collection = Arc::new(Collection::init_from(
            self.path.clone(),
            latest_version,
            toc,
            segments,
            collection_config,
        )?);
        Ok(collection)
    }
}

// TODO(hicder): Add tests once I write builder and writer for SPANN.
#[cfg(test)]
mod tests {
    use anyhow::Result;
    use config::collection::CollectionConfig;
    use quantization::noq::noq::NoQuantizerL2;
    use tempdir::TempDir;
    use utils::test_utils::generate_random_vector;

    use super::*;
    use crate::collection::snapshot::Snapshot;
    use crate::multi_spann::builder::MultiSpannBuilder;
    use crate::multi_spann::writer::MultiSpannWriter;

    fn collection_config() -> CollectionConfig {
        CollectionConfig::default_test_config()
    }

    fn create_segment(base_directory: String) -> Result<()> {
        let num_vectors = 1000;
        let num_features = 4;
        let collection_config = collection_config();
        let mut builder =
            MultiSpannBuilder::new(collection_config, base_directory.clone()).unwrap();

        // Generate 1000 vectors of f32, dimension 4
        for i in 0..num_vectors {
            builder
                .insert(
                    (i % 5) as u128,
                    i as u128,
                    &generate_random_vector(num_features),
                )
                .unwrap();
        }
        builder.build().unwrap();
        let spann_writer = MultiSpannWriter::new(base_directory.clone());
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

        let reader = CollectionReader::new(base_directory.clone());
        let collection = reader.read().unwrap();

        // Check current version
        assert_eq!(collection.current_version(), 1);

        // Get current snapshot
        let snapshot: Snapshot<NoQuantizerL2> = collection.get_snapshot().unwrap();
        assert_eq!(snapshot.segments.len(), 2);
    }
}
