use std::sync::Arc;

use anyhow::{Ok, Result};
use serde::{Deserialize, Serialize};

use super::Collection;
use crate::collection::BoxedSegmentSearchable;
use crate::segment::immutable_segment::ImmutableSegment;
use crate::spann::reader::SpannReader;

pub struct Reader {
    path: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct Manifest {
    segments: Vec<String>,
}

impl Reader {
    pub fn new(path: String) -> Self {
        Self { path }
    }

    pub fn read(&self) -> Result<Arc<Collection>> {
        // Read the manifest file
        let manifest_path = format!("{}/manifest.json", self.path);
        let manifest: Manifest = serde_json::from_reader(std::fs::File::open(manifest_path)?)?;

        let collection = Arc::new(Collection::new());
        let mut segments: Vec<Arc<BoxedSegmentSearchable>> = vec![];
        for name in &manifest.segments {
            let spann_path = format!("{}/{}", self.path, name);
            let spann_reader = SpannReader::new(spann_path);
            let index = spann_reader.read()?;
            segments.push(Arc::new(Box::new(ImmutableSegment::new(index))));
        }

        collection.add_segments(manifest.segments.clone(), segments);
        Ok(collection)
    }
}

// TODO(hicder): Add tests once I write builder and writer for SPANN.
