pub mod immutable_segment;
pub mod mutable_segment;
pub mod pending_segment;

use std::sync::Arc;

use anyhow::Result;
use immutable_segment::ImmutableSegment;
use parking_lot::{Mutex, RwLock};
use pending_segment::PendingSegment;
use quantization::quantization::Quantizer;

use crate::spann::iter::SpannIter;
use crate::utils::IdWithScore;
use crate::vector::StorageContext;

/// A segment is a partial index: users can insert some documents, then flush
/// the containing collection, to effectively create a segment.
pub trait Segment {
    /// Inserts a document into the segment.
    /// Returns an error if the insertion process fails for any reason.
    /// NOTE: Some type of segment may not support insertion.
    fn insert(&self, doc_id: u128, data: &[f32]) -> Result<()>;

    /// Removes a document for an user from the segment.
    /// Returns true if the document was removed, false if the document was not found.
    /// Returns an error if the removal process fails for any reason.
    fn remove(&self, user_id: u128, doc_id: u128) -> Result<bool>;

    /// Returns true if the segment may contain the given document.
    /// False if the segment definitely does not contain the document.
    fn may_contains(&self, doc_id: u128) -> bool;

    /// Returns the name of the segment.
    fn name(&self) -> String;
}

#[derive(Clone)]
pub enum BoxedImmutableSegment<Q: Quantizer + Clone + Send + Sync> {
    FinalizedSegment(Arc<RwLock<ImmutableSegment<Q>>>),
    PendingSegment(Arc<RwLock<PendingSegment<Q>>>),

    // For tests
    MockedNoQuantizationSegment(Arc<RwLock<MockedSegment>>),
}

impl<Q: Quantizer + Clone + Send + Sync> BoxedImmutableSegment<Q> {
    pub fn user_ids(&self) -> Vec<u128> {
        match self {
            BoxedImmutableSegment::FinalizedSegment(immutable_segment) => {
                immutable_segment.read().user_ids()
            }
            BoxedImmutableSegment::PendingSegment(pending_segment) => {
                pending_segment.read().all_user_ids()
            }
            BoxedImmutableSegment::MockedNoQuantizationSegment(mocked_segment) => {
                mocked_segment.read().ids_to_return.clone()
            }
        }
    }

    pub fn iter_for_user(&self, user_id: u128) -> Option<SpannIter<Q>> {
        match self {
            BoxedImmutableSegment::FinalizedSegment(immutable_segment) => {
                immutable_segment.read().iter_for_user(user_id)
            }
            _ => None,
        }
    }

    /// Only get the size of the index from immutable segments for now
    pub fn size_in_bytes_immutable_segments(&self) -> u64 {
        match self {
            BoxedImmutableSegment::FinalizedSegment(immutable_segment) => {
                immutable_segment.read().size_in_bytes()
            }
            BoxedImmutableSegment::PendingSegment(_pending_segment) => 0,
            BoxedImmutableSegment::MockedNoQuantizationSegment(_mocked_segment) => 0,
        }
    }
}

impl<Q: Quantizer + Clone + Send + Sync + 'static> BoxedImmutableSegment<Q> {
    pub fn search_with_id(
        s: BoxedImmutableSegment<Q>,
        id: u128,
        query: Vec<f32>,
        k: usize,
        ef_construction: u32,
        context: Arc<Mutex<impl StorageContext + Send + Sync + 'static>>,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Option<Vec<IdWithScore>>> + Send + Sync>,
    >
    where
        <Q as Quantizer>::QuantizedT: Send + Sync,
    {
        Box::pin(async move {
            match s {
                BoxedImmutableSegment::FinalizedSegment(immutable_segment) => {
                    immutable_segment
                        .read()
                        .search_with_id(id, query.clone(), k, ef_construction, context)
                        .await
                }
                BoxedImmutableSegment::PendingSegment(pending_segment) => {
                    pending_segment
                        .read()
                        .search_with_id(id, query.clone(), k, ef_construction, context)
                        .await
                }
                BoxedImmutableSegment::MockedNoQuantizationSegment(mocked_segment) => {
                    mocked_segment
                        .read()
                        .search_with_id(id, query.clone(), k, ef_construction, context)
                        .await
                }
            }
        })
    }
}

impl<Q: Quantizer + Clone + Send + Sync> Segment for BoxedImmutableSegment<Q> {
    fn insert(&self, doc_id: u128, data: &[f32]) -> Result<()> {
        match self {
            BoxedImmutableSegment::FinalizedSegment(immutable_segment) => {
                immutable_segment.write().insert(doc_id, data)
            }
            BoxedImmutableSegment::PendingSegment(pending_segment) => {
                pending_segment.write().insert(doc_id, data)
            }
            BoxedImmutableSegment::MockedNoQuantizationSegment(mocked_segment) => {
                mocked_segment.write().insert(doc_id, data)
            }
        }
    }

    fn remove(&self, user_id: u128, doc_id: u128) -> Result<bool> {
        match self {
            BoxedImmutableSegment::FinalizedSegment(immutable_segment) => {
                immutable_segment.read().remove(user_id, doc_id)
            }
            BoxedImmutableSegment::PendingSegment(pending_segment) => {
                pending_segment.read().remove(user_id, doc_id)
            }
            BoxedImmutableSegment::MockedNoQuantizationSegment(mocked_segment) => {
                mocked_segment.read().remove(user_id, doc_id)
            }
        }
    }

    fn may_contains(&self, doc_id: u128) -> bool {
        match self {
            BoxedImmutableSegment::FinalizedSegment(immutable_segment) => {
                immutable_segment.read().may_contains(doc_id)
            }
            BoxedImmutableSegment::PendingSegment(pending_segment) => {
                pending_segment.read().may_contains(doc_id)
            }
            BoxedImmutableSegment::MockedNoQuantizationSegment(mocked_segment) => {
                mocked_segment.read().may_contains(doc_id)
            }
        }
    }

    fn name(&self) -> String {
        match self {
            BoxedImmutableSegment::FinalizedSegment(immutable_segment) => {
                immutable_segment.read().name()
            }
            BoxedImmutableSegment::PendingSegment(pending_segment) => pending_segment.read().name(),
            BoxedImmutableSegment::MockedNoQuantizationSegment(mocked_segment) => {
                mocked_segment.read().name()
            }
        }
    }
}

unsafe impl<Q: Quantizer + Clone + Send + Sync> Send for BoxedImmutableSegment<Q> {}
unsafe impl<Q: Quantizer + Clone + Send + Sync> Sync for BoxedImmutableSegment<Q> {}

pub struct MockedSegment {
    name: String,
    ids_to_return: Vec<u128>,
}

impl MockedSegment {
    pub fn new(name: String) -> Self {
        Self {
            name,
            ids_to_return: vec![],
        }
    }

    pub fn set_ids_to_return(&mut self, ids_to_return: Vec<u128>) {
        self.ids_to_return = ids_to_return;
    }
}

#[allow(unused)]
impl MockedSegment {
    pub fn search(
        &self,
        query: Vec<f32>,
        k: usize,
        ef_construction: u32,
        context: Arc<Mutex<impl StorageContext>>,
    ) -> Option<Vec<crate::utils::IdWithScore>> {
        todo!()
    }

    pub async fn search_with_id(
        &self,
        id: u128,
        query: Vec<f32>,
        k: usize,
        ef_construction: u32,
        context: Arc<Mutex<impl StorageContext>>,
    ) -> Option<Vec<crate::utils::IdWithScore>> {
        todo!()
    }
}

#[allow(unused)]
impl Segment for MockedSegment {
    fn insert(&self, doc_id: u128, data: &[f32]) -> Result<()> {
        todo!()
    }

    fn remove(&self, user_id: u128, doc_id: u128) -> Result<bool> {
        todo!()
    }

    fn may_contains(&self, doc_id: u128) -> bool {
        todo!()
    }

    fn name(&self) -> String {
        self.name.clone()
    }
}
