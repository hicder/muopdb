use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use log::info;
use proto::aggregator::aggregator_server::Aggregator;
use proto::aggregator::{GetRequest, GetResponse};
use proto::muopdb::index_server_client::IndexServerClient;
use proto::muopdb::SearchRequest;
use tokio::sync::RwLock;

use crate::node_manager::{self, NodeManager};
use crate::shard_manager::ShardManager;

pub struct AggregatorServerImpl {
    shard_manager: Arc<RwLock<ShardManager>>,
    node_manager: Arc<RwLock<NodeManager>>,
}

impl AggregatorServerImpl {
    pub fn new(
        shard_manager: Arc<RwLock<ShardManager>>,
        node_manager: Arc<RwLock<NodeManager>>,
    ) -> Self {
        Self {
            shard_manager,
            node_manager,
        }
    }
}

struct IdAndScore {
    low_id: u64,
    high_id: u64,
    score: f32,
}

#[tonic::async_trait]
impl Aggregator for AggregatorServerImpl {
    async fn get(
        &self,
        request: tonic::Request<GetRequest>,
    ) -> Result<tonic::Response<GetResponse>, tonic::Status> {
        info!("Got a request: {:?}", request);
        let req = request.into_inner();
        let index_name = &req.index;
        let shard_nodes = self
            .shard_manager
            .read()
            .await
            .get_nodes_for_index(index_name)
            .await
            .ok_or_else(|| {
                tonic::Status::internal(format!("No nodes found for index: {}", index_name))
            })?;
        let ef_construction = req.ef_construction;
        let low_user_ids = req.low_user_ids;
        let high_user_ids = req.high_user_ids;

        let node_infos = self
            .node_manager
            .read()
            .await
            .get_nodes(
                &shard_nodes
                    .iter()
                    .map(|x| x.node_id)
                    .collect::<HashSet<u32>>(),
            )
            .await;
        let node_id_to_node_info: HashMap<u32, node_manager::NodeInfo> =
            node_infos.iter().map(|x| (x.node_id, x.clone())).collect();

        let mut vecs_and_scores: Vec<IdAndScore> = vec![];
        let mut num_pages_accessed: usize = 0;

        // TODO: parallelize
        for shard_node in shard_nodes.iter() {
            let node_info = node_id_to_node_info
                .get(&shard_node.node_id)
                .ok_or_else(|| {
                    tonic::Status::internal(format!(
                        "Node info not found for node_id: {}",
                        shard_node.node_id
                    ))
                })?;
            let mut client =
                IndexServerClient::connect(format!("{}:{}", node_info.ip, node_info.port))
                    .await
                    .map_err(|e| {
                        tonic::Status::internal(format!("Failed to connect to index server: {}", e))
                    })?;

            let index_name_for_shard = format!("{}--{}", index_name, shard_node.shard_id);
            let ret = client
                .search(tonic::Request::new(SearchRequest {
                    collection_name: index_name_for_shard,
                    vector: req.vector.clone(),
                    top_k: req.top_k,
                    record_metrics: req.record_metrics,
                    ef_construction,
                    low_user_ids: low_user_ids.clone(),
                    high_user_ids: high_user_ids.clone(),
                }))
                .await
                .map_err(|e| tonic::Status::internal(format!("Search request failed: {}", e)))?;

            let inner = ret.into_inner();
            inner
                .low_ids
                .iter()
                .zip(inner.high_ids.iter())
                .zip(inner.scores.iter())
                .for_each(|(id, score)| {
                    vecs_and_scores.push(IdAndScore {
                        low_id: *id.0,
                        high_id: *id.1,
                        score: *score,
                    });
                    num_pages_accessed += inner.num_pages_accessed as usize;
                });
        }

        // Sort by score
        vecs_and_scores.sort_by(|a, b| b.score.total_cmp(&a.score));

        Ok(tonic::Response::new(GetResponse {
            low_ids: vecs_and_scores.iter().map(|x| x.low_id).collect(),
            high_ids: vecs_and_scores.iter().map(|x| x.high_id).collect(),
            num_pages_accessed: num_pages_accessed as u64,
        }))
    }
}
