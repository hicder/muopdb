use std::sync::Arc;
use std::vec;

use config::attribute_schema::AttributeSchema;
use config::collection::CollectionConfig;
use config::search_params::SearchParams;
use index::collection::snapshot::SnapshotWithQuantizer;
use index::wal::entry::WalOpType;
use log::info;
use metrics::API_METRICS;
use proto::muopdb::index_server_server::IndexServer;
use proto::muopdb::{
    CreateCollectionRequest, CreateCollectionResponse, DocumentAttribute, FlushRequest,
    FlushResponse, InsertPackedRequest, InsertPackedResponse, InsertRequest, InsertResponse,
    RemoveRequest, RemoveResponse, SearchRequest, SearchResponse,
};
use tokio::sync::RwLock;
use utils::mem::{bytes_to_u128s, ids_to_u128s, transmute_u8_to_slice, u128_to_id};

use crate::collection_manager::CollectionManager;

pub struct IndexServerImpl {
    collection_manager: Arc<RwLock<CollectionManager>>,
}

impl IndexServerImpl {
    pub fn new(collection_manager: Arc<RwLock<CollectionManager>>) -> Self {
        Self { collection_manager }
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
            collection_config.wal_file_size = wal_file_size;
        }
        if let Some(max_pending_ops) = req.max_pending_ops {
            collection_config.max_pending_ops = max_pending_ops;
        }
        if let Some(max_time_to_flush_ms) = req.max_time_to_flush_ms {
            collection_config.max_time_to_flush_ms = max_time_to_flush_ms;
        }
        if let Some(max_number_of_segments) = req.max_number_of_segments {
            collection_config.max_number_of_segments = max_number_of_segments as usize;
        }

        if let Some(proto_schema) = req.attribute_schema {
            collection_config.attribute_schema = Some(AttributeSchema::from(proto_schema));
        }

        let mut collection_manager_locked = self.collection_manager.write().await;
        if collection_manager_locked
            .collection_exists(&collection_name)
            .await
        {
            return Err(tonic::Status::new(
                tonic::Code::AlreadyExists,
                format!("Collection {collection_name} already exists"),
            ));
        }
        match collection_manager_locked
            .add_collection(collection_name.clone(), collection_config)
            .await
        {
            Ok(_) => {
                let duration = start.elapsed();
                info!("[{collection_name}] Created collection in {duration:?}");

                API_METRICS
                    .request_latency_ms_observe("create_collection", duration.as_millis() as f64);

                return Ok(tonic::Response::new(CreateCollectionResponse {
                    message: format!("Collection {collection_name} created"),
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
        let user_ids = ids_to_u128s(&req.user_ids)
            .map_err(|e| tonic::Status::invalid_argument(format!("Invalid user_ids: {}", e)))?;
        let where_document = req.where_document;

        let collection_opt = self
            .collection_manager
            .read()
            .await
            .get_collection(&collection_name)
            .await;
        let filter = if let Some(filter) = where_document {
            Some(Arc::new(filter))
        } else {
            None
        };

        let params = req
            .params
            .ok_or_else(|| tonic::Status::invalid_argument("params is required"))?;
        let search_params = SearchParams::new(
            params.top_k as usize,
            params.ef_construction,
            params.record_metrics,
        )
        .with_num_explored_centroids(params.num_explored_centroids.map(|v| v as usize))
        .with_centroid_distance_ratio(params.centroid_distance_ratio);

        if let Some(collection) = collection_opt {
            if let Ok(snapshot) = collection.get_snapshot().await {
                let result = SnapshotWithQuantizer::search_for_users(
                    snapshot,
                    &user_ids,
                    vec,
                    &search_params,
                    filter,
                )
                .await;

                match result {
                    Some(result) => {
                        let mut doc_ids = vec![];
                        let mut scores = vec![];
                        for id_with_score in result.id_with_scores {
                            // TODO(hicder): Support u128
                            doc_ids.push(u128_to_id(id_with_score.doc_id));

                            scores.push(id_with_score.score);
                        }
                        let end = std::time::Instant::now();
                        let duration = end.duration_since(start);
                        info!("[{collection_name}] Searched collection in {duration:?}");

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

        let doc_ids = ids_to_u128s(&req.doc_ids)
            .map_err(|e| tonic::Status::invalid_argument(format!("Invalid doc_ids: {}", e)))?;
        let vectors = req.vectors;
        let user_ids = ids_to_u128s(&req.user_ids)
            .map_err(|e| tonic::Status::invalid_argument(format!("Invalid user_ids: {}", e)))?;
        let collection_opt = self
            .collection_manager
            .read()
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

                let doc_ids: Arc<[u128]> = Arc::from(doc_ids);
                let user_ids: Arc<[u128]> = Arc::from(user_ids);
                let vectors: Arc<[f32]> = Arc::from(vectors);

                let doc_attrs: Option<Arc<[Option<DocumentAttribute>]>> =
                    req.attributes.as_ref().map(|attrs| {
                        Arc::from(attrs.values.iter().cloned().map(Some).collect::<Vec<_>>())
                    });

                let seq_no = collection
                    .write_to_wal(
                        doc_ids.clone(),
                        user_ids.clone(),
                        WalOpType::Insert(vectors.clone()),
                        doc_attrs,
                    )
                    .await
                    .unwrap_or(0);
                let num_docs_inserted = doc_ids.len() as u32;
                info!(
                    "[{collection_name}] Inserted {num_docs_inserted} vectors in WAL with seq_no {seq_no}"
                );

                if collection.use_wal() {
                    let latency_ms = start.elapsed().as_millis() as f64;
                    API_METRICS.request_latency_ms_observe("insert", latency_ms);

                    return Ok(tonic::Response::new(InsertResponse { num_docs_inserted }));
                }

                let doc_attrs = req.attributes.as_ref().map_or_else(
                    || vec![None; doc_ids.len()],
                    |attrs| attrs.values.iter().cloned().map(Some).collect::<Vec<_>>(),
                );

                for ((vector, &id), doc_attr) in vectors
                    .chunks(dimensions)
                    .zip(doc_ids.iter())
                    .zip(doc_attrs)
                {
                    collection
                        .insert_for_users(&user_ids, id, vector, seq_no, doc_attr)
                        .await
                        .unwrap();
                }

                // log the duration
                let end = std::time::Instant::now();
                let duration = end.duration_since(start);
                info!("[{collection_name}] Inserted {num_docs_inserted} vectors in {duration:?}");

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

        let ids = ids_to_u128s(&req.doc_ids)
            .map_err(|e| tonic::Status::invalid_argument(format!("Invalid doc_ids: {}", e)))?;
        let user_ids = ids_to_u128s(&req.user_ids)
            .map_err(|e| tonic::Status::invalid_argument(format!("Invalid user_ids: {}", e)))?;
        let collection_opt = self
            .collection_manager
            .read()
            .await
            .get_collection(&collection_name)
            .await;

        match collection_opt {
            Some(collection) => {
                let ids: Arc<[u128]> = Arc::from(ids);
                let user_ids: Arc<[u128]> = Arc::from(user_ids);

                let seq_no = collection
                    .write_to_wal(ids.clone(), user_ids.clone(), WalOpType::Delete, None)
                    .await
                    .unwrap_or(0);
                let num_docs_removed = ids.len() as u32;
                info!(
                    "[{collection_name}] Removed {num_docs_removed} vectors from WAL with seq_no {seq_no}"
                );

                let success = true;
                if collection.use_wal() {
                    let latency_ms = start.elapsed().as_millis() as f64;
                    API_METRICS.request_latency_ms_observe("remove", latency_ms);

                    return Ok(tonic::Response::new(RemoveResponse { success }));
                }

                for &user_id in user_ids.iter() {
                    for &doc_id in ids.iter() {
                        collection.remove(user_id, doc_id, seq_no).await.unwrap();
                    }
                }

                // log the duration
                let end = std::time::Instant::now();
                let duration = end.duration_since(start);
                info!("[{collection_name}] Removed {num_docs_removed} vectors in {duration:?}");

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
            .collection_manager
            .read()
            .await
            .get_collection(&collection_name)
            .await;

        let end = std::time::Instant::now();

        match collection_opt {
            Some(collection) => {
                let flushed_segment = collection.flush().await.unwrap();
                let duration = end.duration_since(start);
                info!("Flushed collection {collection_name} in {duration:?}");

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
        let user_ids = ids_to_u128s(&req.user_ids)
            .map_err(|e| tonic::Status::invalid_argument(format!("Invalid user_ids: {}", e)))?;

        let collection_opt = self
            .collection_manager
            .read()
            .await
            .get_collection(&collection_name)
            .await;

        match collection_opt {
            Some(collection) => {
                let dimensions = collection.dimensions();
                let vectors = transmute_u8_to_slice::<f32>(&vectors_buffer);

                if vectors.len() % dimensions != 0 {
                    return Err(tonic::Status::new(
                        tonic::Code::InvalidArgument,
                        "Vectors must be a multiple of the number of dimensions",
                    ));
                }

                let doc_ids: Arc<[u128]> = Arc::from(doc_ids);
                let user_ids: Arc<[u128]> = Arc::from(user_ids);
                let vectors: Arc<[f32]> = Arc::from(vectors);

                let doc_attrs: Option<Arc<[Option<DocumentAttribute>]>> =
                    req.attributes.as_ref().map(|attrs| {
                        Arc::from(attrs.values.iter().cloned().map(Some).collect::<Vec<_>>())
                    });

                let seq_no = collection
                    .write_to_wal(
                        doc_ids.clone(),
                        user_ids.clone(),
                        WalOpType::Insert(vectors.clone()),
                        doc_attrs,
                    )
                    .await
                    .unwrap_or(0);
                let num_docs_inserted = doc_ids.len() as u32;
                info!("Inserted {num_docs_inserted} vectors in WAL with seq_no {seq_no}");

                if collection.use_wal() {
                    let latency_ms = start.elapsed().as_millis() as f64;
                    API_METRICS.request_latency_ms_observe("insert_packed", latency_ms);

                    return Ok(tonic::Response::new(InsertPackedResponse {
                        num_docs_inserted,
                    }));
                }

                let doc_attrs = req.attributes.as_ref().map_or_else(
                    || vec![None; doc_ids.len()],
                    |attrs| attrs.values.iter().cloned().map(Some).collect::<Vec<_>>(),
                );

                for ((vector, &id), doc_attr) in vectors
                    .chunks(dimensions)
                    .zip(doc_ids.iter())
                    .zip(doc_attrs)
                {
                    collection
                        .insert_for_users(&user_ids, id, vector, seq_no, doc_attr)
                        .await
                        .unwrap();
                }

                // log the duration
                let end = std::time::Instant::now();
                let duration = end.duration_since(start);
                info!("[{collection_name}] Inserted {num_docs} vectors in {duration:?}");

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
