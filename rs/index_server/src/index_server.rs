use std::sync::Arc;

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

        let index = self.index_catalog.lock().await.get_index(&index_name).await;
        if let Some(index) = index {
            let result = index.search(&vec, k as usize);
            info!("Search result: {:?}", result);

            match result {
                Some(result) => {
                    let mut ids = vec![];
                    let mut scores = vec![];
                    for id_with_score in result {
                        ids.push(id_with_score.id);
                        scores.push(id_with_score.score);
                    }
                    return Ok(tonic::Response::new(SearchResponse { ids, scores }));
                }
                None => {
                    return Ok(tonic::Response::new(SearchResponse {
                        ids: vec![],
                        scores: vec![],
                    }));
                }
            }
        }
        Ok(tonic::Response::new(SearchResponse {
            ids: vec![],
            scores: vec![],
        }))
    }
}
