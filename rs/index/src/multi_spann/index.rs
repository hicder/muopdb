use std::sync::Arc;

use anyhow::Result;
use dashmap::DashMap;
use memmap2::Mmap;
use odht::HashTableOwned;
use quantization::quantization::Quantizer;

use super::user_index_info::HashConfig;
use crate::index::Searchable;
use crate::spann::index::Spann;
use crate::spann::reader::SpannReader;
use crate::utils::{IdWithScore, SearchContext};

pub struct MultiSpannIndex<Q: Quantizer> {
    base_directory: String,
    user_to_spann: DashMap<u64, Arc<Spann<Q>>>,
    #[allow(dead_code)]
    user_index_info_mmap: Mmap,
    user_index_infos: HashTableOwned<HashConfig>,
}

impl<Q: Quantizer> MultiSpannIndex<Q> {
    pub fn new(base_directory: String, user_index_info_mmap: Mmap) -> Result<Self> {
        let user_index_infos = HashTableOwned::from_raw_bytes(&user_index_info_mmap).unwrap();
        Ok(Self {
            base_directory,
            user_to_spann: DashMap::new(),
            user_index_info_mmap,
            user_index_infos,
        })
    }
}

impl<Q: Quantizer> Searchable for MultiSpannIndex<Q> {
    fn search(
        &self,
        query: &[f32],
        k: usize,
        ef_construction: u32,
        context: &mut SearchContext,
    ) -> Option<Vec<IdWithScore>> {
        self.search_with_id(0, query, k, ef_construction, context)
    }

    fn search_with_id(
        &self,
        id: u64,
        query: &[f32],
        k: usize,
        ef_construction: u32,
        context: &mut SearchContext,
    ) -> Option<Vec<IdWithScore>> {
        let index = self.user_to_spann.get(&id);
        if index.is_none() {
            // Fetch the index from the mmap
            let index_info = self.user_index_infos.get(&id);
            if index_info.is_none() {
                return None;
            }

            let index_info = index_info.unwrap();
            let reader = SpannReader::new_with_offsets(
                self.base_directory.clone(),
                index_info.centroid_index_offset as usize,
                index_info.centroid_vector_offset as usize,
                index_info.ivf_index_offset as usize,
                index_info.ivf_vectors_offset as usize,
            );
            match reader.read::<Q>() {
                Ok(index) => {
                    let index = Arc::new(index);
                    self.user_to_spann.insert(id, index.clone());
                    return index.search(query, k, ef_construction, context);
                }
                Err(_) => {
                    return None;
                }
            }
        }

        let index = index.unwrap().clone();
        index.search(query, k, ef_construction, context)
    }
}
