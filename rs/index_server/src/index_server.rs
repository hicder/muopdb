use std::sync::Arc;
use std::vec;

use index::utils::SearchContext;
use log::{debug, info};
use proto::muopdb::index_server_server::IndexServer;
use proto::muopdb::{
    FlushRequest, FlushResponse, InsertPackedRequest, InsertPackedResponse, InsertRequest,
    InsertResponse, SearchRequest, SearchResponse,
};
use tokio::sync::Mutex;
use utils::mem::transmute_u8_to_slice;

use crate::collection_catalog::CollectionCatalog;

pub struct IndexServerImpl {
    pub index_catalog: Arc<Mutex<CollectionCatalog>>,
}

impl IndexServerImpl {
    pub fn new(index_catalog: Arc<Mutex<CollectionCatalog>>) -> Self {
        Self { index_catalog }
    }
}

#[tonic::async_trait]
impl IndexServer for IndexServerImpl {
    async fn search(
        &self,
        request: tonic::Request<SearchRequest>,
    ) -> Result<tonic::Response<SearchResponse>, tonic::Status> {
        let start = std::time::Instant::now();
        let req = request.into_inner();
        let index_name = req.index_name;
        let vec = req.vector;
        let k = req.top_k;
        let record_metrics = req.record_metrics;
        let ef_construction = req.ef_construction;
        let user_ids = req.user_ids;

        let collection_opt = self
            .index_catalog
            .lock()
            .await
            .get_collection(&index_name)
            .await;
        if let Some(collection) = collection_opt {
            let mut search_context = SearchContext::new(record_metrics);
            if let Ok(snapshot) = collection.get_snapshot() {
                let result = snapshot.search_for_ids(
                    &user_ids,
                    &vec,
                    k as usize,
                    ef_construction,
                    &mut search_context,
                );
                info!("Search result: {:?}", result);

                match result {
                    Some(result) => {
                        let mut ids = vec![];
                        let mut scores = vec![];
                        for id_with_score in result {
                            ids.push(id_with_score.id);
                            scores.push(id_with_score.score);
                        }
                        let end = std::time::Instant::now();
                        let duration = end.duration_since(start);
                        debug!("Searched collection {} in {:?}", index_name, duration);
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
            } else {
                return Err(tonic::Status::new(
                    tonic::Code::Internal,
                    "Failed to get snapshot",
                ));
            }
        }
        Err(tonic::Status::new(
            tonic::Code::NotFound,
            "Collection not found",
        ))
    }

    async fn insert(
        &self,
        request: tonic::Request<InsertRequest>,
    ) -> Result<tonic::Response<InsertResponse>, tonic::Status> {
        let start = std::time::Instant::now();
        let req = request.into_inner();
        let collection_name = req.collection_name;
        let ids = req.ids;
        let vectors = req.vectors;
        let user_ids = req.user_ids;

        let collection_opt = self
            .index_catalog
            .lock()
            .await
            .get_collection(&collection_name)
            .await;

        match collection_opt {
            Some(collection) => {
                let dimensions = collection.dimensions();
                if vectors.len() % dimensions != 0 {
                    return Err(tonic::Status::new(
                        tonic::Code::InvalidArgument,
                        "Vectors must be a multiple of the number of dimensions",
                    ));
                }

                vectors
                    .chunks(dimensions)
                    .zip(&ids)
                    .for_each(|(vector, id)| {
                        // TODO(hicder): Handle errors
                        collection.insert_for_users(&user_ids, *id, vector).unwrap()
                    });

                // log the duration
                let end = std::time::Instant::now();
                let duration = end.duration_since(start);
                debug!("Inserted {} vectors in {:?}", ids.len(), duration);
                Ok(tonic::Response::new(InsertResponse { inserted_ids: ids }))
            }
            None => Err(tonic::Status::new(
                tonic::Code::NotFound,
                "Collection not found",
            )),
        }
    }

    async fn flush(
        &self,
        request: tonic::Request<FlushRequest>,
    ) -> Result<tonic::Response<FlushResponse>, tonic::Status> {
        let start = std::time::Instant::now();
        let req = request.into_inner();
        let collection_name = req.collection_name;

        let collection_opt = self
            .index_catalog
            .lock()
            .await
            .get_collection(&collection_name)
            .await;

        let end = std::time::Instant::now();
        let duration = end.duration_since(start);
        debug!("Indexing collection {} in {:?}", collection_name, duration);

        match collection_opt {
            Some(collection) => {
                collection.flush().unwrap();
                Ok(tonic::Response::new(FlushResponse {
                    // TODO(hicder): Return flushed segments
                    flushed_segments: vec![],
                }))
            }
            None => Err(tonic::Status::new(
                tonic::Code::NotFound,
                "Collection not found",
            )),
        }
    }

    async fn insert_packed(
        &self,
        request: tonic::Request<InsertPackedRequest>,
    ) -> Result<tonic::Response<InsertPackedResponse>, tonic::Status> {
        let start = std::time::Instant::now();
        let req = request.into_inner();
        let collection_name = req.collection_name;
        let ids_buffer = req.ids;
        let vectors_buffer = req.vectors;
        let user_ids = req.user_ids;

        let collection_opt = self
            .index_catalog
            .lock()
            .await
            .get_collection(&collection_name)
            .await;

        match collection_opt {
            Some(collection) => {
                let dimensions = collection.dimensions();
                let vectors = transmute_u8_to_slice(&vectors_buffer);
                let ids = transmute_u8_to_slice(&ids_buffer);

                if vectors.len() % dimensions != 0 {
                    return Err(tonic::Status::new(
                        tonic::Code::InvalidArgument,
                        "Vectors must be a multiple of the number of dimensions",
                    ));
                }

                vectors
                    .chunks(dimensions)
                    .zip(ids)
                    .for_each(|(vector, id)| {
                        // TODO(hicder): Handle errors
                        collection.insert_for_users(&user_ids, *id, vector).unwrap()
                    });

                // log the duration
                let end = std::time::Instant::now();
                let duration = end.duration_since(start);
                debug!("Inserted {} vectors in {:?}", ids.len(), duration);
                Ok(tonic::Response::new(InsertPackedResponse {}))
            }
            None => Err(tonic::Status::new(
                tonic::Code::NotFound,
                "Collection not found",
            )),
        }
    }
}
