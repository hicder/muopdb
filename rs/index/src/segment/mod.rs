pub mod immutable_segment;
pub mod mutable_segment;
pub mod pending_segment;

use std::sync::Arc;

use anyhow::Result;
use immutable_segment::ImmutableSegment;
use parking_lot::RwLock;
use pending_segment::PendingSegment;
use quantization::noq::noq::NoQuantizerL2;
use quantization::pq::pq::ProductQuantizerL2;

use crate::index::Searchable;
use crate::utils::{IdWithScore, SearchContext};

/// A segment is a partial index: users can insert some documents, then flush
/// the containing collection, to effectively create a segment.
pub trait Segment {
    /// Inserts a document into the segment.
    /// Returns an error if the insertion process fails for any reason.
    /// NOTE: Some type of segment may not support insertion.
    fn insert(&self, doc_id: u64, data: &[f32]) -> Result<()>;

    /// Removes a document from the segment.
    /// Returns true if the document was removed, false if the document was not found.
    /// Returns an error if the removal process fails for any reason.
    fn remove(&self, doc_id: u64) -> Result<bool>;

    /// Returns true if the segment may contain the given document.
    /// False if the segment definitely does not contain the document.
    fn may_contains(&self, doc_id: u64) -> bool;

    /// Returns the name of the segment.
    fn name(&self) -> String;
}

// TODO(hicder): Add different types of distance
#[derive(Clone)]
pub enum BoxedImmutableSegment {
    FinalizedNoQuantizationSegment(Arc<RwLock<ImmutableSegment<NoQuantizerL2>>>),
    FinalizedProductQuantizationSegment(Arc<RwLock<ImmutableSegment<ProductQuantizerL2>>>),

    PendingNoQuantizationSegment(Arc<RwLock<PendingSegment<NoQuantizerL2>>>),
    PendingProductQuantizationSegment(Arc<RwLock<PendingSegment<ProductQuantizerL2>>>),

    // For tests
    MockedNoQuantizationSegment(Arc<RwLock<MockedSegment>>),
}

impl Searchable for BoxedImmutableSegment {
    fn search(
        &self,
        query: &[f32],
        k: usize,
        ef_construction: u32,
        context: &mut crate::utils::SearchContext,
    ) -> Option<Vec<crate::utils::IdWithScore>> {
        self.search_with_id(0u128, query, k, ef_construction, context)
    }

    fn search_with_id(
        &self,
        id: u128,
        query: &[f32],
        k: usize,
        ef_construction: u32,
        context: &mut SearchContext,
    ) -> Option<Vec<IdWithScore>> {
        match self {
            BoxedImmutableSegment::FinalizedNoQuantizationSegment(immutable_segment) => {
                immutable_segment
                    .read()
                    .search_with_id(id, query, k, ef_construction, context)
            }
            BoxedImmutableSegment::FinalizedProductQuantizationSegment(immutable_segment) => {
                immutable_segment
                    .read()
                    .search_with_id(id, query, k, ef_construction, context)
            }
            BoxedImmutableSegment::MockedNoQuantizationSegment(mocked_segment) => mocked_segment
                .read()
                .search_with_id(id, query, k, ef_construction, context),
            BoxedImmutableSegment::PendingNoQuantizationSegment(pending_segment) => pending_segment
                .read()
                .search_with_id(id, query, k, ef_construction, context),
            BoxedImmutableSegment::PendingProductQuantizationSegment(pending_segment) => {
                pending_segment
                    .read()
                    .search_with_id(id, query, k, ef_construction, context)
            }
        }
    }
}

impl Segment for BoxedImmutableSegment {
    fn insert(&self, doc_id: u64, data: &[f32]) -> Result<()> {
        match self {
            BoxedImmutableSegment::FinalizedNoQuantizationSegment(immutable_segment) => {
                immutable_segment.write().insert(doc_id, data)
            }
            BoxedImmutableSegment::FinalizedProductQuantizationSegment(immutable_segment) => {
                immutable_segment.write().insert(doc_id, data)
            }
            BoxedImmutableSegment::MockedNoQuantizationSegment(mocked_segment) => {
                mocked_segment.write().insert(doc_id, data)
            }
            BoxedImmutableSegment::PendingNoQuantizationSegment(pending_segment) => {
                pending_segment.write().insert(doc_id, data)
            }
            BoxedImmutableSegment::PendingProductQuantizationSegment(pending_segment) => {
                pending_segment.write().insert(doc_id, data)
            }
        }
    }

    fn remove(&self, doc_id: u64) -> Result<bool> {
        match self {
            BoxedImmutableSegment::FinalizedNoQuantizationSegment(immutable_segment) => {
                immutable_segment.read().remove(doc_id)
            }
            BoxedImmutableSegment::FinalizedProductQuantizationSegment(immutable_segment) => {
                immutable_segment.read().remove(doc_id)
            }
            BoxedImmutableSegment::MockedNoQuantizationSegment(mocked_segment) => {
                mocked_segment.read().remove(doc_id)
            }
            BoxedImmutableSegment::PendingNoQuantizationSegment(pending_segment) => {
                pending_segment.read().remove(doc_id)
            }
            BoxedImmutableSegment::PendingProductQuantizationSegment(pending_segment) => {
                pending_segment.read().remove(doc_id)
            }
        }
    }

    fn may_contains(&self, doc_id: u64) -> bool {
        match self {
            BoxedImmutableSegment::FinalizedNoQuantizationSegment(immutable_segment) => {
                immutable_segment.read().may_contains(doc_id)
            }
            BoxedImmutableSegment::FinalizedProductQuantizationSegment(immutable_segment) => {
                immutable_segment.read().may_contains(doc_id)
            }
            BoxedImmutableSegment::MockedNoQuantizationSegment(mocked_segment) => {
                mocked_segment.read().may_contains(doc_id)
            }
            BoxedImmutableSegment::PendingNoQuantizationSegment(pending_segment) => {
                pending_segment.read().may_contains(doc_id)
            }
            BoxedImmutableSegment::PendingProductQuantizationSegment(pending_segment) => {
                pending_segment.read().may_contains(doc_id)
            }
        }
    }

    fn name(&self) -> String {
        match self {
            BoxedImmutableSegment::FinalizedNoQuantizationSegment(immutable_segment) => {
                immutable_segment.read().name()
            }
            BoxedImmutableSegment::FinalizedProductQuantizationSegment(immutable_segment) => {
                immutable_segment.read().name()
            }
            BoxedImmutableSegment::PendingNoQuantizationSegment(pending_segment) => {
                pending_segment.read().name()
            }
            BoxedImmutableSegment::PendingProductQuantizationSegment(pending_segment) => {
                pending_segment.read().name()
            }
            BoxedImmutableSegment::MockedNoQuantizationSegment(mocked_segment) => {
                mocked_segment.read().name()
            }
        }
    }
}

unsafe impl Send for BoxedImmutableSegment {}
unsafe impl Sync for BoxedImmutableSegment {}

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
impl Searchable for MockedSegment {
    fn search(
        &self,
        query: &[f32],
        k: usize,
        ef_construction: u32,
        context: &mut crate::utils::SearchContext,
    ) -> Option<Vec<crate::utils::IdWithScore>> {
        todo!()
    }

    fn search_with_id(
        &self,
        id: u128,
        query: &[f32],
        k: usize,
        ef_construction: u32,
        context: &mut crate::utils::SearchContext,
    ) -> Option<Vec<crate::utils::IdWithScore>> {
        todo!()
    }
}

#[allow(unused)]
impl Segment for MockedSegment {
    fn insert(&self, doc_id: u64, data: &[f32]) -> Result<()> {
        todo!()
    }

    fn remove(&self, doc_id: u64) -> Result<bool> {
        todo!()
    }

    fn may_contains(&self, doc_id: u64) -> bool {
        todo!()
    }

    fn name(&self) -> String {
        self.name.clone()
    }
}
