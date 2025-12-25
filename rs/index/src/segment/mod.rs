pub mod immutable_segment;
pub mod mutable_segment;
pub mod pending_mutable_segment;
pub mod pending_segment;

use std::sync::Arc;

use anyhow::Result;
use async_lock::RwLock;
use config::search_params::SearchParams;
use immutable_segment::ImmutableSegment;
use pending_segment::PendingSegment;
use quantization::quantization::Quantizer;

use crate::multi_terms::index::MultiTermIndex;
use crate::query::planner::Planner;
use crate::spann::iter::SpannIter;
use crate::utils::SearchResult;

/// A segment is a partial index: users can insert some documents, then flush
/// the containing collection, to effectively create a segment.
#[async_trait::async_trait]
pub trait Segment {
    /// Inserts a document into the segment.
    /// Returns an error if the insertion process fails for any reason.
    /// NOTE: Some type of segment may not support insertion.
    async fn insert(&self, doc_id: u128, data: &[f32]) -> Result<()>;

    /// Removes a document for an user from the segment.
    /// Returns true if the document was removed, false if the document was not found.
    /// Returns an error if the removal process fails for any reason.
    async fn remove(&self, user_id: u128, doc_id: u128) -> Result<bool>;

    /// Returns true if the segment may contain the given document.
    /// False if the segment definitely does not contain the document.
    async fn may_contain(&self, doc_id: u128) -> bool;

    /// Returns the name of the segment.
    async fn name(&self) -> String;
}

#[derive(Clone)]
pub enum BoxedImmutableSegment<Q: Quantizer + Clone + Send + Sync> {
    FinalizedSegment(Arc<RwLock<ImmutableSegment<Q>>>),
    PendingSegment(Arc<RwLock<PendingSegment<Q>>>),

    // For tests
    MockedNoQuantizationSegment(Arc<RwLock<MockedSegment>>),
}

impl<Q: Quantizer + Clone + Send + Sync> BoxedImmutableSegment<Q> {
    pub async fn user_ids(&self) -> Vec<u128> {
        match self {
            BoxedImmutableSegment::FinalizedSegment(immutable_segment) => {
                immutable_segment.read().await.user_ids()
            }
            BoxedImmutableSegment::PendingSegment(pending_segment) => {
                pending_segment.read().await.all_user_ids()
            }
            BoxedImmutableSegment::MockedNoQuantizationSegment(mocked_segment) => {
                mocked_segment.read().await.ids_to_return.clone()
            }
        }
    }

    pub async fn iter_for_user(&self, user_id: u128) -> Option<SpannIter<Q>> {
        match self {
            BoxedImmutableSegment::FinalizedSegment(immutable_segment) => {
                immutable_segment.read().await.iter_for_user(user_id).await
            }
            _ => None,
        }
    }

    pub async fn is_invalidated(&self, user_id: u128, doc_id: u128) -> Result<bool> {
        match self {
            BoxedImmutableSegment::FinalizedSegment(immutable_segment) => {
                immutable_segment
                    .read()
                    .await
                    .is_invalidated(user_id, doc_id)
                    .await
            }
            _ => Ok(false),
        }
    }

    /// Only get the size of the index from immutable segments for now
    pub async fn size_in_bytes_immutable_segments(&self) -> u64 {
        match self {
            BoxedImmutableSegment::FinalizedSegment(immutable_segment) => {
                immutable_segment.read().await.size_in_bytes()
            }
            BoxedImmutableSegment::PendingSegment(_pending_segment) => 0,
            BoxedImmutableSegment::MockedNoQuantizationSegment(_mocked_segment) => 0,
        }
    }

    pub async fn num_docs(&self) -> u64 {
        match self {
            BoxedImmutableSegment::FinalizedSegment(immutable_segment) => {
                immutable_segment.read().await.num_docs().unwrap_or(0) as u64
            }
            BoxedImmutableSegment::PendingSegment(_pending_segment) => 0,
            BoxedImmutableSegment::MockedNoQuantizationSegment(_mocked_segment) => 0,
        }
    }

    pub async fn should_auto_vacuum(&self) -> bool {
        match self {
            BoxedImmutableSegment::FinalizedSegment(immutable_segment) => {
                immutable_segment.read().await.should_auto_vacuum().await
            }
            BoxedImmutableSegment::PendingSegment(_pending_segment) => false,
            BoxedImmutableSegment::MockedNoQuantizationSegment(_mocked_segment) => false,
        }
    }

    pub async fn get_multi_term_index(&self) -> Option<Arc<MultiTermIndex>> {
        match self {
            BoxedImmutableSegment::FinalizedSegment(immutable_segment) => {
                immutable_segment.read().await.get_multi_term_index()
            }
            _ => None,
        }
    }
}

#[allow(clippy::await_holding_lock)]
impl<Q: Quantizer + Clone + Send + Sync + 'static> BoxedImmutableSegment<Q> {
    pub fn search_with_id<'a>(
        s: BoxedImmutableSegment<Q>,
        id: u128,
        query: Vec<f32>,
        params: &'a SearchParams,
        planner: Option<Arc<Planner>>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Option<SearchResult>> + Send + 'a>>
    where
        <Q as Quantizer>::QuantizedT: Send + Sync,
    {
        Box::pin(async move {
            match s {
                BoxedImmutableSegment::FinalizedSegment(immutable_segment) => {
                    immutable_segment
                        .read()
                        .await
                        .search_for_user(id, query.clone(), params, planner)
                        .await
                }
                BoxedImmutableSegment::PendingSegment(pending_segment) => {
                    pending_segment
                        .read()
                        .await
                        .search_with_id(id, query.clone(), params)
                        .await
                }
                BoxedImmutableSegment::MockedNoQuantizationSegment(mocked_segment) => {
                    mocked_segment
                        .read()
                        .await
                        .search_with_id(id, query.clone(), params)
                        .await
                }
            }
        })
    }
}

#[async_trait::async_trait]
impl<Q: Quantizer + Clone + Send + Sync> Segment for BoxedImmutableSegment<Q> {
    async fn insert(&self, doc_id: u128, data: &[f32]) -> Result<()> {
        match self {
            BoxedImmutableSegment::FinalizedSegment(immutable_segment) => {
                immutable_segment.write().await.insert(doc_id, data).await
            }
            BoxedImmutableSegment::PendingSegment(pending_segment) => {
                pending_segment.write().await.insert(doc_id, data).await
            }
            BoxedImmutableSegment::MockedNoQuantizationSegment(mocked_segment) => {
                mocked_segment.write().await.insert(doc_id, data).await
            }
        }
    }

    async fn remove(&self, user_id: u128, doc_id: u128) -> Result<bool> {
        match self {
            BoxedImmutableSegment::FinalizedSegment(immutable_segment) => {
                immutable_segment.read().await.remove(user_id, doc_id).await
            }
            BoxedImmutableSegment::PendingSegment(pending_segment) => {
                pending_segment.read().await.remove(user_id, doc_id).await
            }
            BoxedImmutableSegment::MockedNoQuantizationSegment(mocked_segment) => {
                mocked_segment.read().await.remove(user_id, doc_id).await
            }
        }
    }

    async fn may_contain(&self, doc_id: u128) -> bool {
        match self {
            BoxedImmutableSegment::FinalizedSegment(immutable_segment) => {
                immutable_segment.read().await.may_contain(doc_id).await
            }
            BoxedImmutableSegment::PendingSegment(pending_segment) => {
                pending_segment.read().await.may_contain(doc_id).await
            }
            BoxedImmutableSegment::MockedNoQuantizationSegment(mocked_segment) => {
                mocked_segment.read().await.may_contain(doc_id).await
            }
        }
    }

    async fn name(&self) -> String {
        match self {
            BoxedImmutableSegment::FinalizedSegment(immutable_segment) => {
                immutable_segment.read().await.name().await
            }
            BoxedImmutableSegment::PendingSegment(pending_segment) => {
                pending_segment.read().await.name().await
            }
            BoxedImmutableSegment::MockedNoQuantizationSegment(mocked_segment) => {
                mocked_segment.read().await.name().await
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
        record_pages: bool,
    ) -> Option<SearchResult> {
        todo!()
    }

    pub async fn search_with_id(
        &self,
        id: u128,
        query: Vec<f32>,
        params: &SearchParams,
    ) -> Option<SearchResult> {
        todo!()
    }
}

#[allow(unused)]
#[async_trait::async_trait]
impl Segment for MockedSegment {
    async fn insert(&self, doc_id: u128, data: &[f32]) -> Result<()> {
        todo!()
    }

    async fn remove(&self, user_id: u128, doc_id: u128) -> Result<bool> {
        todo!()
    }

    async fn may_contain(&self, doc_id: u128) -> bool {
        todo!()
    }

    async fn name(&self) -> String {
        self.name.clone()
    }
}
