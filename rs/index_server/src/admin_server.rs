use std::sync::Arc;

use index::collection::BoxedCollection;
use index::optimizers::engine::{OptimizerEngine, OptimizingType};
use log::info;
use proto::admin::index_server_admin_server::IndexServerAdmin;
use proto::admin::{
    GetSegmentsRequest, GetSegmentsResponse, MergeSegmentsRequest, MergeSegmentsResponse,
    SegmentInfo,
};
use tokio::sync::RwLock;

use crate::collection_manager::CollectionManager;

pub struct AdminServerImpl {
    collection_manager: Arc<RwLock<CollectionManager>>,
}

impl AdminServerImpl {
    pub fn new(collection_manager: Arc<RwLock<CollectionManager>>) -> Self {
        Self { collection_manager }
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
            .collection_manager
            .read()
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
                "Collection {collection_name} not found"
            )));
        }

        Ok(tonic::Response::new(MergeSegmentsResponse {
            segment_name: returned_segment_name,
        }))
    }

    async fn get_segments(
        &self,
        request: tonic::Request<GetSegmentsRequest>,
    ) -> Result<tonic::Response<GetSegmentsResponse>, tonic::Status> {
        let start = std::time::Instant::now();
        let req = request.into_inner();
        let collection_name = req.collection_name;

        let collection_opt = self
            .collection_manager
            .read()
            .await
            .get_collection(&collection_name)
            .await;

        match collection_opt {
            Some(collection) => {
                let segment_infos = collection.get_active_segment_infos();
                let returned_segment_infos = segment_infos
                    .segment_infos
                    .iter()
                    .map(|segment_info| SegmentInfo {
                        segment_name: segment_info.name.clone(),
                        size_in_bytes: segment_info.size_in_bytes,
                    })
                    .collect();
                let end = std::time::Instant::now();
                let duration = end.duration_since(start);
                info!("[{collection_name}] Get segments in {duration:?}");

                Ok(tonic::Response::new(GetSegmentsResponse {
                    segment_infos: returned_segment_infos,
                    version: segment_infos.version,
                }))
            }
            None => Err(tonic::Status::new(
                tonic::Code::NotFound,
                "Collection not found",
            )),
        }
    }
}
