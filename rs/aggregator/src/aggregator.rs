use log::info;
use proto::muopdb::aggregator_server::Aggregator;
use proto::muopdb::{GetRequest, GetResponse};

pub struct AggregatorServerImpl {}

#[tonic::async_trait]
impl Aggregator for AggregatorServerImpl {
    async fn get(
        &self,
        request: tonic::Request<GetRequest>,
    ) -> Result<tonic::Response<GetResponse>, tonic::Status> {
        info!("Got a request: {:?}", request);
        Ok(tonic::Response::new(GetResponse {
            value: "Hello, world!".to_string(),
        }))
    }
}
