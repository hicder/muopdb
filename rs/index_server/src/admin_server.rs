use std::sync::Arc;

use index::collection::BoxedCollection;
use index::optimizers::engine::{OptimizerEngine, OptimizingType};
use proto::admin::index_server_admin_server::IndexServerAdmin;
use proto::admin::{MergeSegmentsRequest, MergeSegmentsResponse};
use tokio::sync::Mutex;

use crate::collection_catalog::CollectionCatalog;

pub struct AdminServerImpl {
    pub collection_catalog: Arc<Mutex<CollectionCatalog>>,
}

impl AdminServerImpl {
    pub fn new(collection_catalog: Arc<Mutex<CollectionCatalog>>) -> Self {
        Self { collection_catalog }
    }
}

#[tonic::async_trait]
impl IndexServerAdmin for AdminServerImpl {
    async fn merge_segments(
        &self,
        _request: tonic::Request<MergeSegmentsRequest>,
    ) -> Result<tonic::Response<MergeSegmentsResponse>, tonic::Status> {
        let req = _request.into_inner();
        let collection_name = req.collection_name;
        let segment_names = req.segment_names;
        let returned_segment_name;
        if let Some(collection) = self
            .collection_catalog
            .lock()
            .await
            .get_collection(&collection_name)
            .await
        {
            match collection {
                BoxedCollection::CollectionNoQuantizationL2(collection) => {
                    let optimizer = OptimizerEngine::new(collection);
                    match optimizer.run(segment_names, OptimizingType::Merge) {
                        Ok(new_segment_name) => {
                            returned_segment_name = new_segment_name;
                        }
                        Err(e) => {
                            return Err(tonic::Status::internal(e.to_string()));
                        }
                    }
                }
                BoxedCollection::CollectionProductQuantization(_collection) => {
                    // We don't support merging collection with product quantization yet
                    return Err(tonic::Status::unimplemented(
                        "Product quantization is not supported yet",
                    ));
                }
            }
        } else {
            return Err(tonic::Status::not_found(format!(
                "Collection {} not found",
                collection_name
            )));
        }

        Ok(tonic::Response::new(MergeSegmentsResponse {
            segment_name: returned_segment_name,
        }))
    }
}
