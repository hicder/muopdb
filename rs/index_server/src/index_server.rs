use std::sync::Arc;
use std::vec;

use config::collection::CollectionConfig;
use index::utils::SearchContext;
use log::info;
use proto::muopdb::index_server_server::IndexServer;
use proto::muopdb::{
    CreateCollectionRequest, CreateCollectionResponse, FlushRequest, FlushResponse,
    GetSegmentsRequest, GetSegmentsResponse, InsertPackedRequest, InsertPackedResponse,
    InsertRequest, InsertResponse, SearchRequest, SearchResponse, CompactSegmentsRequest, CompactSegmentsResponse,
};
use tokio::sync::Mutex;
use utils::mem::{lows_and_highs_to_u128s, transmute_u8_to_slice, u128s_to_lows_highs};

use crate::collection_catalog::CollectionCatalog;
use crate::collection_manager::CollectionManager;

pub struct IndexServerImpl {
    pub collection_catalog: Arc<Mutex<CollectionCatalog>>,
    pub collection_manager: Arc<Mutex<CollectionManager>>,
}

impl IndexServerImpl {
    pub fn new(
        index_catalog: Arc<Mutex<CollectionCatalog>>,
        collection_manager: Arc<Mutex<CollectionManager>>,
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
        let mut collection_config = CollectionConfig::default();
        let req = request.into_inner();
        let collection_name = req.collection_name;

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

        let mut collection_manager_locked = self.collection_manager.lock().await;
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
                return Ok(tonic::Response::new(CreateCollectionResponse {}));
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
        let vec = req.vector;
        let k = req.top_k;
        let record_metrics = req.record_metrics;
        let ef_construction = req.ef_construction;
        let user_ids = lows_and_highs_to_u128s(&req.low_user_ids, &req.high_user_ids);

        let collection_opt = self
            .collection_catalog
            .lock()
            .await
            .get_collection(&collection_name)
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

                match result {
                    Some(result) => {
                        let mut low_ids = vec![];
                        let mut high_ids = vec![];
                        let mut scores = vec![];
                        for id_with_score in result {
                            // TODO(hicder): Support u128
                            low_ids.push(id_with_score.id as u64);
                            high_ids.push((id_with_score.id >> 64) as u64);
                            scores.push(id_with_score.score);
                        }
                        let end = std::time::Instant::now();
                        let duration = end.duration_since(start);
                        info!(
                            "[{}] Searched collection in {:?}",
                            collection_name, duration
                        );
                        return Ok(tonic::Response::new(SearchResponse {
                            low_ids,
                            high_ids,
                            scores,
                            num_pages_accessed: search_context.num_pages_accessed() as u64,
                        }));
                    }
                    None => {
                        return Ok(tonic::Response::new(SearchResponse {
                            low_ids: vec![],
                            high_ids: vec![],
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
        let ids = lows_and_highs_to_u128s(&req.low_ids, &req.high_ids);
        let vectors = req.vectors;
        let user_ids = lows_and_highs_to_u128s(&req.low_user_ids, &req.high_user_ids);
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
                info!(
                    "[{}] Inserted {} vectors in {:?}",
                    collection_name,
                    ids.len(),
                    duration
                );

                let lows_and_highs = u128s_to_lows_highs(&ids);
                Ok(tonic::Response::new(InsertResponse {
                    inserted_low_ids: lows_and_highs.lows,
                    inserted_high_ids: lows_and_highs.highs,
                }))
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
            .collection_catalog
            .lock()
            .await
            .get_collection(&collection_name)
            .await;

        let end = std::time::Instant::now();

        match collection_opt {
            Some(collection) => {
                collection.flush().unwrap();
                let duration = end.duration_since(start);
                info!("Flushed collection {} in {:?}", collection_name, duration);
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
        let doc_ids = lows_and_highs_to_u128s(
            transmute_u8_to_slice(&req.low_ids),
            transmute_u8_to_slice(&req.high_ids),
        );
        let num_docs = doc_ids.len();
        let vectors_buffer = req.vectors;
        let user_ids = lows_and_highs_to_u128s(&req.low_user_ids, &req.high_user_ids);

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

                vectors
                    .chunks(dimensions)
                    .zip(doc_ids)
                    .for_each(|(vector, id)| {
                        // TODO(hicder): Handle errors
                        collection.insert_for_users(&user_ids, id, vector).unwrap()
                    });

                // log the duration
                let end = std::time::Instant::now();
                let duration = end.duration_since(start);
                info!(
                    "[{}] Inserted {} vectors in {:?}",
                    collection_name, num_docs, duration
                );
                Ok(tonic::Response::new(InsertPackedResponse {}))
            }
            None => Err(tonic::Status::new(
                tonic::Code::NotFound,
                "Collection not found",
            )),
        }
    }

    async fn get_segments(
        &self,
        request: tonic::Request<GetSegmentsRequest>,
    ) -> Result<tonic::Response<GetSegmentsResponse>, tonic::Status> {
        let start = std::time::Instant::now();
        let req = request.into_inner();
        let collection_name = req.collection_name;

        let collection_opt = self
            .collection_catalog
            .lock()
            .await
            .get_collection(&collection_name)
            .await;

        match collection_opt {
            Some(collection) => {
                let segments = collection.get_all_segment_names();
                let end = std::time::Instant::now();
                let duration = end.duration_since(start);
                info!("[{}] Get segments in {:?}", collection_name, duration);

                Ok(tonic::Response::new(GetSegmentsResponse {
                    segment_names: segments,
                }))
            }
            None => Err(tonic::Status::new(
                tonic::Code::NotFound,
                "Collection not found",
            )),
        }
    }

    async fn compact_segments(
        &self,
        request: tonic::Request<CompactSegmentsRequest>,
    ) -> Result<tonic::Response<CompactSegmentsResponse>, tonic::Status> {
        let start = std::time::Instant::now();
        let req = request.into_inner();
        let collection_name = req.collection_name;
        let segment_names = req.segment_names;

        let collection_opt = self
            .collection_catalog
            .lock()
            .await
            .get_collection(&collection_name)
            .await;

        match collection_opt {
            Some(collection) => {
                // Validation that segments exist in the collection
                let segments = collection.get_all_segment_names();
                let missing_segments: Vec<String> = segment_names
                    .iter()
                    .filter(|segment_name| !segments.contains(segment_name))
                    .cloned()
                    .collect();
                if !missing_segments.is_empty() {
                    return Err(tonic::Status::new(
                        tonic::Code::NotFound,
                        format!("Segments not found: {:?}", missing_segments),
                    ));
                }

                if segment_names.len() <= 1 {
                    return Err(tonic::Status::new(
                        tonic::Code::InvalidArgument,
                        "Require at least 2 segments to compact",
                    ));
                }

                // TODO- khoa165: Logic to compact segments here
                
                let end = std::time::Instant::now();
                let duration = end.duration_since(start);
                info!("[{}] Compacted {} segments in {:?}", collection_name, segment_names.len(), duration);

                Ok(tonic::Response::new(CompactSegmentsResponse {}))
            }
            None => Err(tonic::Status::new(
                tonic::Code::NotFound,
                "Collection not found",
            )),
        }
    }
}
