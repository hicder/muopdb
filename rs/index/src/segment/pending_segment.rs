use std::collections::HashSet;
use std::path::PathBuf;

use anyhow::{Ok, Result};
use config::collection::CollectionConfig;
use parking_lot::RwLock;
use quantization::quantization::Quantizer;

use super::{BoxedImmutableSegment, Segment};
use crate::multi_spann::index::MultiSpannIndex;
use crate::multi_spann::reader::MultiSpannReader;
use crate::utils::{IdWithScore, SearchContext};

pub struct PendingSegment<Q: Quantizer + Clone> {
    inner_segments: Vec<BoxedImmutableSegment<Q>>,
    inner_segments_names: Vec<String>,
    name: String,
    parent_directory: String,

    // Whether to use the internal index instead of passing the query to inner segments.
    // Invariant: use_internal_index is true if and only if index is Some.
    use_internal_index: bool,

    // The internal index.
    index: RwLock<Option<MultiSpannIndex<Q>>>,

    collection_config: CollectionConfig,
}

impl<Q: Quantizer + Clone> PendingSegment<Q> {
    pub fn new(
        inner_segments: Vec<BoxedImmutableSegment<Q>>,
        data_directory: String,
        collection_config: CollectionConfig,
    ) -> Self {
        let path = PathBuf::from(&data_directory);
        // name is the last portion of the data_directory
        let name = path.file_name().unwrap().to_str().unwrap().to_string();

        // base directory is the directory of the data_directory
        let parent_directory = path.parent().unwrap().to_str().unwrap().to_string();
        let inner_segments_names = inner_segments
            .iter()
            .map(|segment| segment.name())
            .collect();

        Self {
            inner_segments,
            inner_segments_names,
            name,
            parent_directory,
            index: RwLock::new(None),
            use_internal_index: false,
            collection_config,
        }
    }

    // Caller must hold the read lock before calling this function.
    pub fn build_index(&self) -> Result<()> {
        if self.use_internal_index {
            // We shouldn't build the index if it already exists.
            return Err(anyhow::anyhow!("Index already exists"));
        }

        let current_directory = format!("{}/{}", self.parent_directory, self.name);
        let reader = MultiSpannReader::new(current_directory);
        let index = reader.read::<Q>()?;
        self.index.write().replace(index);
        Ok(())
    }

    // Caller must hold the write lock before calling this function.
    pub fn apply_pending_deletions(&self) -> Result<()> {
        // TODO(hicder): Implement this once we support deletions.
        Ok(())
    }

    // Caller must hold the write lock before calling this function.
    pub fn switch_to_internal_index(&mut self) {
        self.use_internal_index = true;
    }

    pub fn inner_segments_names(&self) -> &Vec<String> {
        &self.inner_segments_names
    }

    pub fn parent_directory(&self) -> &String {
        &self.parent_directory
    }

    pub fn base_directory(&self) -> String {
        format!("{}/{}", self.parent_directory, self.name)
    }

    pub fn inner_segments(&self) -> &Vec<BoxedImmutableSegment<Q>> {
        &self.inner_segments
    }

    pub fn collection_config(&self) -> &CollectionConfig {
        &self.collection_config
    }

    pub fn all_user_ids(&self) -> Vec<u128> {
        let mut user_ids = HashSet::new();
        for segment in &self.inner_segments {
            user_ids.extend(segment.user_ids());
        }
        user_ids.into_iter().collect()
    }
}

#[allow(unused)]
impl<Q: Quantizer + Clone> Segment for PendingSegment<Q> {
    fn insert(&self, doc_id: u128, data: &[f32]) -> Result<()> {
        Err(anyhow::anyhow!("Pending segment does not support insert"))
    }

    fn remove(&self, user_id: u128, doc_id: u128) -> Result<bool> {
        todo!()
    }

    fn may_contains(&self, _doc_id: u128) -> bool {
        todo!()
    }

    fn name(&self) -> String {
        self.name.clone()
    }
}

impl<Q: Quantizer + Clone> PendingSegment<Q> {
    pub fn search_with_id(
        &self,
        id: u128,
        query: &[f32],
        k: usize,
        ef_construction: u32,
        context: &mut SearchContext,
    ) -> Option<Vec<IdWithScore>> {
        if !self.use_internal_index {
            let mut results = Vec::new();
            for segment in &self.inner_segments {
                let segment_result = segment.search_with_id(id, query, k, ef_construction, context);
                if let Some(result) = segment_result {
                    results.extend(result);
                }
            }
            Some(results)
        } else {
            let index = self.index.read();
            match &*index {
                Some(index) => index.search_with_id(id, query, k, ef_construction, context),
                None => None,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use config::collection::CollectionConfig;
    use quantization::noq::noq::{NoQuantizer, NoQuantizerL2};
    use rand::Rng;
    use utils::distance::l2::L2DistanceCalculator;

    use super::*;
    use crate::multi_spann::builder::MultiSpannBuilder;
    use crate::multi_spann::writer::MultiSpannWriter;
    use crate::segment::immutable_segment::ImmutableSegment;

    fn build_segment(base_directory: String, starting_doc_id: u128) -> Result<()> {
        let mut starting_doc_id = starting_doc_id;
        let mut spann_builder_config = CollectionConfig::default_test_config();
        spann_builder_config.num_features = 4;
        let mut multi_spann_builder =
            MultiSpannBuilder::new(spann_builder_config, base_directory.clone())?;

        multi_spann_builder.insert(starting_doc_id % 2, starting_doc_id, &[1.0, 2.0, 3.0, 4.0])?;
        starting_doc_id += 1;
        multi_spann_builder.insert(starting_doc_id % 2, starting_doc_id, &[5.0, 6.0, 7.0, 8.0])?;
        starting_doc_id += 1;
        multi_spann_builder.insert(
            starting_doc_id % 2,
            starting_doc_id,
            &[9.0, 10.0, 11.0, 12.0],
        )?;
        multi_spann_builder.build()?;

        let multi_spann_writer = MultiSpannWriter::new(base_directory.clone());
        multi_spann_writer.write(&mut multi_spann_builder)?;
        Ok(())
    }

    fn read_segment(base_directory: String) -> Result<MultiSpannIndex<NoQuantizerL2>> {
        let reader = MultiSpannReader::new(base_directory);
        let index = reader.read::<NoQuantizerL2>()?;
        Ok(index)
    }

    #[test]
    fn test_pending_segment() -> Result<()> {
        // temp directory
        let tmp_dir = tempdir::TempDir::new("pending_segment_test").unwrap();
        let base_dir = tmp_dir.path().to_str().unwrap().to_string();

        // Create dir for segment1
        let segment1_dir = format!("{}/segment_1", base_dir);
        std::fs::create_dir_all(segment1_dir.clone()).unwrap();
        build_segment(segment1_dir.clone(), 0)?;
        let segment1 = read_segment(segment1_dir.clone())?;
        let segment1 =
            BoxedImmutableSegment::<NoQuantizer<L2DistanceCalculator>>::FinalizedSegment(Arc::new(
                RwLock::new(ImmutableSegment::new(segment1, "segment_1".to_string())),
            ));

        let random_name = format!(
            "pending_segment_{}",
            rand::thread_rng().gen_range(0..1000000)
        );
        let pending_dir = format!("{}/{}", base_dir, random_name);
        std::fs::create_dir_all(pending_dir.clone()).unwrap();

        // Create a pending segment
        let pending_segment = PendingSegment::<NoQuantizer<L2DistanceCalculator>>::new(
            vec![segment1],
            pending_dir.clone(),
            CollectionConfig::default_test_config(),
        );

        let mut context = SearchContext::new(false);
        let results = pending_segment.search_with_id(0, &[1.0, 2.0, 3.0, 4.0], 1, 10, &mut context);
        let res = results.unwrap();
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].id, 0);

        Ok(())
    }
}
