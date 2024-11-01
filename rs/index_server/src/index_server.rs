use log::info;
use proto::muopdb::index_server_server::IndexServer;
use proto::muopdb::{SearchRequest, SearchResponse};

pub struct IndexServerImpl {}

#[tonic::async_trait]
impl IndexServer for IndexServerImpl {
    async fn search(
        &self,
        request: tonic::Request<SearchRequest>,
    ) -> Result<tonic::Response<SearchResponse>, tonic::Status> {
        info!("Got a request: {:?}", request);
        Ok(tonic::Response::new(SearchResponse { ids: vec![1, 2, 3] }))
    }
}
