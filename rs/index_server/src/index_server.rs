use std::sync::Arc;

use index::utils::SearchContext;
use log::info;
use proto::muopdb::index_server_server::IndexServer;
use proto::muopdb::{SearchRequest, SearchResponse};
use tokio::sync::Mutex;

use crate::index_catalog::IndexCatalog;

pub struct IndexServerImpl {
    pub index_catalog: Arc<Mutex<IndexCatalog>>,
}

impl IndexServerImpl {
    pub fn new(index_catalog: Arc<Mutex<IndexCatalog>>) -> Self {
        Self { index_catalog }
    }
}

#[tonic::async_trait]
impl IndexServer for IndexServerImpl {
    async fn search(
        &self,
        request: tonic::Request<SearchRequest>,
    ) -> Result<tonic::Response<SearchResponse>, tonic::Status> {
        let req = request.into_inner();
        let index_name = req.index_name;
        let vec = req.vector;
        let k = req.top_k;
        let record_metrics = req.record_metrics;

        let index = self.index_catalog.lock().await.get_index(&index_name).await;
        if let Some(index) = index {
            let mut search_context = SearchContext::new(record_metrics);
            let result = index.search(&vec, k as usize, &mut search_context);
            info!("Search result: {:?}", result);

            match result {
                Some(result) => {
                    let mut ids = vec![];
                    let mut scores = vec![];
                    for id_with_score in result {
                        ids.push(id_with_score.id);
                        scores.push(id_with_score.score);
                    }
                    return Ok(tonic::Response::new(SearchResponse {
                        ids: ids,
                        scores: scores,
                        num_pages_accessed: search_context.num_pages_accessed() as u64,
                    }));
                }
                None => {
                    return Ok(tonic::Response::new(SearchResponse {
                        ids: vec![],
                        scores: vec![],
                        num_pages_accessed: 0,
                    }));
                }
            }
        }
        Ok(tonic::Response::new(SearchResponse {
            ids: vec![],
            scores: vec![],
            num_pages_accessed: 0,
        }))
    }
}
