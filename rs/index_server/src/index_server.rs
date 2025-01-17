use std::sync::Arc;
use std::vec;

use config::collection::CollectionConfig;
use index::utils::SearchContext;
use log::{debug, info};
use proto::muopdb::index_server_server::IndexServer;
use proto::muopdb::{
    CreateCollectionRequest, CreateCollectionResponse, FlushRequest, FlushResponse,
    InsertPackedRequest, InsertPackedResponse, InsertRequest, InsertResponse, SearchRequest,
    SearchResponse,
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
        let user_ids: Vec<u128> = req.user_ids.iter().map(|id| *id as u128).collect();

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
                info!("Search result: {:?}", result);

                match result {
                    Some(result) => {
                        let mut ids = vec![];
                        let mut scores = vec![];
                        for id_with_score in result {
                            // TODO(hicder): Support u128
                            ids.push(id_with_score.id as u64);
                            scores.push(id_with_score.score);
                        }
                        let end = std::time::Instant::now();
                        let duration = end.duration_since(start);
                        debug!("Searched collection {} in {:?}", collection_name, duration);
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
        let ids = req.ids.iter().map(|x| *x as u128).collect::<Vec<u128>>();
        let vectors = req.vectors;
        let user_ids = req
            .user_ids
            .iter()
            .map(|x| *x as u128)
            .collect::<Vec<u128>>();

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
                debug!("Inserted {} vectors in {:?}", ids.len(), duration);

                let lows_and_highs = u128s_to_lows_highs(&ids);
                // TODO(hicder): Support high values
                Ok(tonic::Response::new(InsertResponse {
                    inserted_ids: lows_and_highs.lows,
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
        let user_ids = req
            .user_ids
            .iter()
            .map(|id| *id as u128)
            .collect::<Vec<u128>>();

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
                let ids: &[u64] = transmute_u8_to_slice(&ids_buffer);
                let high_ids = vec![0; ids.len()];
                let doc_ids = lows_and_highs_to_u128s(&ids, &high_ids);

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
