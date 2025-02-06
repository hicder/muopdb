pub mod collection;
pub mod reader;
pub mod snapshot;

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use collection::Collection;
use quantization::noq::noq::NoQuantizer;
use quantization::pq::pq::ProductQuantizer;
use serde::{Deserialize, Serialize};
use snapshot::SnapshotWithQuantizer;
use utils::distance::l2::L2DistanceCalculator;
use utils::mem::{transmute_slice_to_u8, transmute_u8_to_slice};

use crate::index::Searchable;
use crate::segment::Segment;
use crate::wal::entry::WalOpType;

pub trait SegmentSearchable: Searchable + Segment {}
pub type BoxedSegmentSearchable = Box<dyn SegmentSearchable + Send + Sync>;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TableOfContent {
    pub toc: Vec<String>,

    #[serde(default)]
    pub pending: HashMap<String, Vec<String>>,

    #[serde(default = "default_sequence_number")]
    pub sequence_number: i64,
}

fn default_sequence_number() -> i64 {
    -1
}

impl TableOfContent {
    pub fn new(toc: Vec<String>) -> Self {
        Self {
            toc,
            pending: HashMap::new(),
            sequence_number: -1,
        }
    }
}

impl Default for TableOfContent {
    fn default() -> Self {
        Self {
            toc: vec![],
            pending: HashMap::new(),
            sequence_number: -1,
        }
    }
}

pub struct VersionsInfo {
    pub current_version: u64,
    pub version_ref_counts: HashMap<u64, usize>,
}

impl VersionsInfo {
    pub fn new() -> Self {
        let mut version_ref_counts = HashMap::new();
        version_ref_counts.insert(0, 0);
        Self {
            current_version: 0,
            version_ref_counts,
        }
    }
}

pub struct OpChannelEntry {
    pub doc_ids_offset: usize,
    pub doc_ids_length: usize,
    pub user_ids_offset: usize,
    pub user_ids_length: usize,
    pub data_offset: usize,
    pub data_length: usize,

    pub seq_no: u64,
    pub op_type: WalOpType,
    pub buffer: Vec<u8>,
}

impl OpChannelEntry {
    pub fn new(
        doc_ids: &[u128],
        user_ids: &[u128],
        data: &[f32],
        seq_no: u64,
        op_type: WalOpType,
    ) -> Self {
        let mut buffer: Vec<u8> = Vec::new();
        buffer.extend_from_slice(transmute_slice_to_u8(doc_ids));
        buffer.extend_from_slice(transmute_slice_to_u8(user_ids));
        buffer.extend_from_slice(transmute_slice_to_u8(data));

        Self {
            doc_ids_offset: 0,
            doc_ids_length: doc_ids.len() * 16,
            user_ids_offset: doc_ids.len() * 16,
            user_ids_length: user_ids.len() * 16,
            data_offset: doc_ids.len() * 16 + user_ids.len() * 16,
            data_length: data.len() * 4,
            seq_no,
            op_type,
            buffer,
        }
    }

    pub fn doc_ids(&self) -> &[u128] {
        transmute_u8_to_slice(
            &self.buffer[self.doc_ids_offset..self.doc_ids_offset + self.doc_ids_length],
        )
    }

    pub fn user_ids(&self) -> &[u128] {
        transmute_u8_to_slice(
            &self.buffer[self.user_ids_offset..self.user_ids_offset + self.user_ids_length],
        )
    }

    pub fn data(&self) -> &[f32] {
        transmute_u8_to_slice(&self.buffer[self.data_offset..self.data_offset + self.data_length])
    }
}

#[derive(Clone)]
pub enum BoxedCollection {
    CollectionNoQuantizationL2(Arc<Collection<NoQuantizer<L2DistanceCalculator>>>),
    CollectionProductQuantization(Arc<Collection<ProductQuantizer<L2DistanceCalculator>>>),
}

impl BoxedCollection {
    pub fn use_wal(&self) -> bool {
        match self {
            BoxedCollection::CollectionNoQuantizationL2(collection) => collection.use_wal(),
            BoxedCollection::CollectionProductQuantization(collection) => collection.use_wal(),
        }
    }

    pub fn dimensions(&self) -> usize {
        match self {
            BoxedCollection::CollectionNoQuantizationL2(collection) => collection.dimensions(),
            BoxedCollection::CollectionProductQuantization(collection) => collection.dimensions(),
        }
    }

    pub async fn write_to_wal(
        &self,
        doc_ids: &[u128],
        user_ids: &[u128],
        data: &[f32],
    ) -> Result<u64> {
        match self {
            BoxedCollection::CollectionNoQuantizationL2(collection) => {
                collection.write_to_wal(doc_ids, user_ids, data).await
            }
            BoxedCollection::CollectionProductQuantization(collection) => {
                collection.write_to_wal(doc_ids, user_ids, data).await
            }
        }
    }

    pub fn should_auto_flush(&self) -> bool {
        match self {
            BoxedCollection::CollectionNoQuantizationL2(collection) => {
                collection.should_auto_flush()
            }
            BoxedCollection::CollectionProductQuantization(collection) => {
                collection.should_auto_flush()
            }
        }
    }

    pub fn insert_for_users(
        &self,
        user_ids: &[u128],
        doc_id: u128,
        data: &[f32],
        seq_no: u64,
    ) -> Result<()> {
        match self {
            BoxedCollection::CollectionNoQuantizationL2(collection) => {
                collection.insert_for_users(user_ids, doc_id, data, seq_no)
            }
            BoxedCollection::CollectionProductQuantization(collection) => {
                collection.insert_for_users(user_ids, doc_id, data, seq_no)
            }
        }
    }

    pub fn flush(&self) -> Result<String> {
        match self {
            BoxedCollection::CollectionNoQuantizationL2(collection) => collection.flush(),
            BoxedCollection::CollectionProductQuantization(collection) => collection.flush(),
        }
    }

    pub fn get_all_segment_names(&self) -> Vec<String> {
        match self {
            BoxedCollection::CollectionNoQuantizationL2(collection) => {
                collection.get_all_segment_names()
            }
            BoxedCollection::CollectionProductQuantization(collection) => {
                collection.get_all_segment_names()
            }
        }
    }

    pub async fn process_one_op(&self) -> Result<usize> {
        match self {
            BoxedCollection::CollectionNoQuantizationL2(collection) => {
                collection.process_one_op().await
            }
            BoxedCollection::CollectionProductQuantization(collection) => {
                collection.process_one_op().await
            }
        }
    }

    pub fn get_snapshot(&self) -> Result<SnapshotWithQuantizer> {
        match self {
            BoxedCollection::CollectionNoQuantizationL2(collection) => {
                let col = Arc::clone(collection);
                let snapshot = col.get_snapshot()?;
                Ok(SnapshotWithQuantizer::new_with_no_quantizer(snapshot))
            }
            BoxedCollection::CollectionProductQuantization(collection) => {
                let col = Arc::clone(collection);
                let snapshot = col.get_snapshot()?;
                Ok(SnapshotWithQuantizer::new_with_product_quantizer(snapshot))
            }
        }
    }
}
