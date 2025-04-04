use std::sync::Arc;
use std::vec;

use config::collection::CollectionConfig;
use index::collection::snapshot::SnapshotWithQuantizer;
use index::wal::entry::WalOpType;
use log::info;
use metrics::API_METRICS;
use proto::muopdb::index_server_server::IndexServer;
use proto::muopdb::{
    CreateCollectionRequest, CreateCollectionResponse, FlushRequest, FlushResponse, Id,
    InsertPackedRequest, InsertPackedResponse, InsertRequest, InsertResponse, RemoveRequest,
    RemoveResponse, SearchRequest, SearchResponse,
};
use tokio::sync::{Mutex, RwLock};
use utils::mem::{bytes_to_u128s, ids_to_u128s, transmute_u8_to_slice};

use crate::collection_catalog::CollectionCatalog;
use crate::collection_manager::CollectionManager;

pub struct IndexServerImpl {
    pub collection_catalog: Arc<Mutex<CollectionCatalog>>,
    pub collection_manager: Arc<RwLock<CollectionManager>>,
}

impl IndexServerImpl {
    pub fn new(
        index_catalog: Arc<Mutex<CollectionCatalog>>,
        collection_manager: Arc<RwLock<CollectionManager>>,
    ) -> Self {
        Self {
            collection_catalog: index_catalog,
            collection_manager,
        }
    }
}

#[tonic::async_trait]
impl IndexServer for IndexServerImpl {
    async fn create_collection(
        &self,
        request: tonic::Request<CreateCollectionRequest>,
    ) -> Result<tonic::Response<CreateCollectionResponse>, tonic::Status> {
        let start = std::time::Instant::now();
        let mut collection_config = CollectionConfig::default();
        let req = request.into_inner();
        let collection_name = req.collection_name;
        API_METRICS.num_requests_inc("create_collection", &collection_name);

        if let Some(num_features) = req.num_features {
            collection_config.num_features = num_features as usize;
        }
        if let Some(centroids_max_neighbors) = req.centroids_max_neighbors {
            collection_config.centroids_max_neighbors = centroids_max_neighbors as usize;
        }
        if let Some(centroids_max_layers) = req.centroids_max_layers {
            collection_config.centroids_max_layers = centroids_max_layers as u8;
        }
        if let Some(centroids_ef_construction) = req.centroids_ef_construction {
            collection_config.centroids_ef_construction = centroids_ef_construction;
        }
        if let Some(memory_size) = req.centroids_builder_vector_storage_memory_size {
            collection_config.centroids_builder_vector_storage_memory_size = memory_size as usize;
        }
        if let Some(file_size) = req.centroids_builder_vector_storage_file_size {
            collection_config.centroids_builder_vector_storage_file_size = file_size as usize;
        }
        if let Some(quantization_type) = req.quantization_type {
            collection_config.quantization_type = quantization_type.into();
        }
        if let Some(max_iter) = req.product_quantization_max_iteration {
            collection_config.product_quantization_max_iteration = max_iter as usize;
        }
        if let Some(batch_size) = req.product_quantization_batch_size {
            collection_config.product_quantization_batch_size = batch_size as usize;
        }
        if let Some(subvec_dim) = req.product_quantization_subvector_dimension {
            collection_config.product_quantization_subvector_dimension = subvec_dim as usize;
        }
        if let Some(num_bits) = req.product_quantization_num_bits {
            collection_config.product_quantization_num_bits = num_bits as usize;
        }
        if let Some(training_rows) = req.product_quantization_num_training_rows {
            collection_config.product_quantization_num_training_rows = training_rows as usize;
        }
        if let Some(num_centroids) = req.initial_num_centroids {
            collection_config.initial_num_centroids = num_centroids as usize;
        }
        if let Some(data_points) = req.num_data_points_for_clustering {
            collection_config.num_data_points_for_clustering = data_points as usize;
        }
        if let Some(max_clusters) = req.max_clusters_per_vector {
            collection_config.max_clusters_per_vector = max_clusters as usize;
        }
        if let Some(threshold_pct) = req.clustering_distance_threshold_pct {
            collection_config.clustering_distance_threshold_pct = threshold_pct;
        }
        if let Some(encoding_type) = req.posting_list_encoding_type {
            collection_config.posting_list_encoding_type = encoding_type.into();
        }
        if let Some(memory_size) = req.posting_list_builder_vector_storage_memory_size {
            collection_config.posting_list_builder_vector_storage_memory_size =
                memory_size as usize;
        }
        if let Some(file_size) = req.posting_list_builder_vector_storage_file_size {
            collection_config.posting_list_builder_vector_storage_file_size = file_size as usize;
        }
        if let Some(max_size) = req.max_posting_list_size {
            collection_config.max_posting_list_size = max_size as usize;
        }
        if let Some(penalty) = req.posting_list_kmeans_unbalanced_penalty {
            collection_config.posting_list_kmeans_unbalanced_penalty = penalty;
        }
        if let Some(reindex) = req.reindex {
            collection_config.reindex = reindex;
        }
        if let Some(wal_file_size) = req.wal_file_size {
            collection_config.wal_file_size = wal_file_size as u64;
        }
        if let Some(max_pending_ops) = req.max_pending_ops {
            collection_config.max_pending_ops = max_pending_ops as u64;
        }
        if let Some(max_time_to_flush_ms) = req.max_time_to_flush_ms {
            collection_config.max_time_to_flush_ms = max_time_to_flush_ms as u64;
        }

        let mut collection_manager_locked = self.collection_manager.write().await;
        if collection_manager_locked
            .collection_exists(&collection_name)
            .await
        {
            return Err(tonic::Status::new(
                tonic::Code::AlreadyExists,
                format!("Collection {} already exists", collection_name),
            ));
        }
        match collection_manager_locked
            .add_collection(collection_name.clone(), collection_config)
            .await
        {
            Ok(_) => {
                let latency_ms = start.elapsed().as_millis() as f64;
                API_METRICS.request_latency_ms_observe("create_collection", latency_ms);

                return Ok(tonic::Response::new(CreateCollectionResponse {
                    message: format!("Collection {} created", collection_name),
                }));
            }
            Err(e) => {
                return Err(tonic::Status::new(tonic::Code::Internal, e.to_string()));
            }
        }
    }

    async fn search(
        &self,
        request: tonic::Request<SearchRequest>,
    ) -> Result<tonic::Response<SearchResponse>, tonic::Status> {
        let start = std::time::Instant::now();
        let req = request.into_inner();
        let collection_name = req.collection_name;
        API_METRICS.num_requests_inc("search", &collection_name);

        let vec = req.vector;
        let k = req.top_k;
        let record_metrics = req.record_metrics;
        let ef_construction = req.ef_construction;
        let user_ids = ids_to_u128s(&req.user_ids);

        let collection_opt = self
            .collection_catalog
            .lock()
            .await
            .get_collection(&collection_name)
            .await;
        if let Some(collection) = collection_opt {
            if let Ok(snapshot) = collection.get_snapshot() {
                let result = SnapshotWithQuantizer::search_for_users(
                    snapshot,
                    &user_ids,
                    vec,
                    k as usize,
                    ef_construction,
                    record_metrics,
                )
                .await;

                match result {
                    Some(result) => {
                        let mut doc_ids = vec![];
                        let mut scores = vec![];
                        for id_with_score in result.id_with_scores {
                            // TODO(hicder): Support u128
                            doc_ids.push(Id {
                                low_id: id_with_score.doc_id as u64,
                                high_id: (id_with_score.doc_id >> 64) as u64,
                            });

                            scores.push(id_with_score.score);
                        }
                        let end = std::time::Instant::now();
                        let duration = end.duration_since(start);
                        info!(
                            "[{}] Searched collection in {:?}",
                            collection_name, duration
                        );

                        API_METRICS
                            .request_latency_ms_observe("search", duration.as_millis() as f64);

                        return Ok(tonic::Response::new(SearchResponse {
                            doc_ids,
                            scores,
                            num_pages_accessed: result.stats.num_pages_accessed as u64,
                        }));
                    }
                    None => {
                        let duration = start.elapsed().as_millis() as f64;
                        API_METRICS.request_latency_ms_observe("search", duration);

                        return Ok(tonic::Response::new(SearchResponse {
                            doc_ids: vec![],
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
        API_METRICS.num_requests_inc("insert", &collection_name);

        let ids = ids_to_u128s(&req.doc_ids);
        let vectors = req.vectors;
        let user_ids = ids_to_u128s(&req.user_ids);
        let collection_opt = self
            .collection_catalog
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

                let seq_no = collection
                    .write_to_wal(&ids, &user_ids, &vectors, WalOpType::Insert)
                    .await
                    .unwrap_or(0);
                let num_docs_inserted = ids.len() as u32;
                info!(
                    "[{}] Inserted {} vectors in WAL with seq_no {}",
                    collection_name, num_docs_inserted, seq_no
                );

                if collection.use_wal() {
                    let latency_ms = start.elapsed().as_millis() as f64;
                    API_METRICS.request_latency_ms_observe("insert", latency_ms);
                    
                    return Ok(tonic::Response::new(InsertResponse { num_docs_inserted }));
                }

                vectors
                    .chunks(dimensions)
                    .zip(&ids)
                    .for_each(|(vector, id)| {
                        // TODO(hicder): Handle errors
                        collection
                            .insert_for_users(&user_ids, *id, vector, seq_no)
                            .unwrap()
                    });

                // log the duration
                let end = std::time::Instant::now();
                let duration = end.duration_since(start);
                info!(
                    "[{}] Inserted {} vectors in {:?}",
                    collection_name, num_docs_inserted, duration
                );

                API_METRICS.request_latency_ms_observe("insert", duration.as_millis() as f64);

                Ok(tonic::Response::new(InsertResponse { num_docs_inserted }))
            }
            None => Err(tonic::Status::new(
                tonic::Code::NotFound,
                "Collection not found",
            )),
        }
    }

    async fn remove(
        &self,
        request: tonic::Request<RemoveRequest>,
    ) -> Result<tonic::Response<RemoveResponse>, tonic::Status> {
        let start = std::time::Instant::now();
        let req = request.into_inner();
        let collection_name = req.collection_name;
        API_METRICS.num_requests_inc("remove", &collection_name);

        let ids = ids_to_u128s(&req.doc_ids);
        let user_ids = ids_to_u128s(&req.user_ids);
        let collection_opt = self
            .collection_catalog
            .lock()
            .await
            .get_collection(&collection_name)
            .await;

        match collection_opt {
            Some(collection) => {
                let seq_no = collection
                    .write_to_wal(&ids, &user_ids, &[], WalOpType::Delete)
                    .await
                    .unwrap_or(0);
                let num_docs_removed = ids.len() as u32;
                info!(
                    "[{}] Removed {} vectors from WAL with seq_no {}",
                    collection_name, num_docs_removed, seq_no
                );

                let success = true;
                if collection.use_wal() {
                    let latency_ms = start.elapsed().as_millis() as f64;
                    API_METRICS.request_latency_ms_observe("remove", latency_ms);

                    return Ok(tonic::Response::new(RemoveResponse { success }));
                }

                user_ids.iter().for_each(|&user_id| {
                    ids.iter().for_each(|&doc_id| {
                        collection.remove(user_id, doc_id, seq_no).unwrap();
                    })
                });

                // log the duration
                let end = std::time::Instant::now();
                let duration = end.duration_since(start);
                info!(
                    "[{}] Removed {} vectors in {:?}",
                    collection_name, num_docs_removed, duration
                );

                API_METRICS.request_latency_ms_observe("remove", duration.as_millis() as f64);

                Ok(tonic::Response::new(RemoveResponse { success }))
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
        API_METRICS.num_requests_inc("flush", &collection_name);

        let collection_opt = self
            .collection_catalog
            .lock()
            .await
            .get_collection(&collection_name)
            .await;

        let end = std::time::Instant::now();

        match collection_opt {
            Some(collection) => {
                let flushed_segment = collection.flush().unwrap();
                let duration = end.duration_since(start);
                info!("Flushed collection {} in {:?}", collection_name, duration);

                API_METRICS.request_latency_ms_observe("flush", duration.as_millis() as f64);

                Ok(tonic::Response::new(FlushResponse {
                    flushed_segments: vec![flushed_segment],
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
        API_METRICS.num_requests_inc("insert_packed", &collection_name);

        let doc_ids = bytes_to_u128s(&req.doc_ids);
        let num_docs = doc_ids.len();
        let vectors_buffer = req.vectors;
        let user_ids = ids_to_u128s(&req.user_ids);

        let collection_opt = self
            .collection_catalog
            .lock()
            .await
            .get_collection(&collection_name)
            .await;

        match collection_opt {
            Some(collection) => {
                let dimensions = collection.dimensions();
                let vectors = transmute_u8_to_slice(&vectors_buffer);

                if vectors.len() % dimensions != 0 {
                    return Err(tonic::Status::new(
                        tonic::Code::InvalidArgument,
                        "Vectors must be a multiple of the number of dimensions",
                    ));
                }

                let seq_no = collection
                    .write_to_wal(&doc_ids, &user_ids, &vectors, WalOpType::Insert)
                    .await
                    .unwrap_or(0);
                let num_docs_inserted = doc_ids.len() as u32;
                info!(
                    "Inserted {} vectors in WAL with seq_no {}",
                    num_docs_inserted, seq_no
                );

                if collection.use_wal() {
                    let latency_ms = start.elapsed().as_millis() as f64;
                    API_METRICS.request_latency_ms_observe("insert_packed", latency_ms);
                    
                    return Ok(tonic::Response::new(InsertPackedResponse {
                        num_docs_inserted,
                    }));
                }

                vectors
                    .chunks(dimensions)
                    .zip(doc_ids)
                    .for_each(|(vector, id)| {
                        // TODO(hicder): Handle errors
                        collection
                            .insert_for_users(&user_ids, id, vector, seq_no)
                            .unwrap()
                    });

                // log the duration
                let end = std::time::Instant::now();
                let duration = end.duration_since(start);
                info!(
                    "[{}] Inserted {} vectors in {:?}",
                    collection_name, num_docs, duration
                );

                API_METRICS
                    .request_latency_ms_observe("insert_packed", duration.as_millis() as f64);

                Ok(tonic::Response::new(InsertPackedResponse {
                    num_docs_inserted,
                }))
            }
            None => Err(tonic::Status::new(
                tonic::Code::NotFound,
                "Collection not found",
            )),
        }
    }
}
