use serde::{Deserialize, Serialize};

use crate::attribute_schema::AttributeSchema;
use crate::enums::{IntSeqEncodingType, QuantizerType};

/// Config for a collection.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct CollectionConfig {
    /// Number of dimensions of the vectors. You'd want to modify this parameter, depending on the
    /// dimensionality of your vectors.
    /// Default: 768
    pub num_features: usize,

    /// For centroids graph. The graph will be stored in HNSW format.
    /// Max number of neighbors to build
    /// Default: 10
    pub centroids_max_neighbors: usize,

    /// Maximum number of layers in the centroids graph.
    /// Default: 10
    pub centroids_max_layers: u8,

    /// ef_construction parameter for HNSW index.
    /// For more details, see https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md
    /// Default: 100
    pub centroids_ef_construction: u32,

    /// Specify the size of the memory in bytes for the HNSW temporary vector storage
    /// in the builder. If the size exceeds this value, data will be spilled to disk.
    /// Default: 1024 * 1024 * 1024 (1MB)
    pub centroids_builder_vector_storage_memory_size: usize,

    /// Specify the size of the file in bytes for the HNSW temporary vector storage file.
    /// in the builder. If the size exceeds this value, a new temporary file will be created.
    /// Default: 1024 * 1024 * 1024 (1MB)
    pub centroids_builder_vector_storage_file_size: usize,

    /// Quantization type. Use as a lossy vector storage compression.
    /// Product Quantization will reduce memory usage, at the cost of recall.
    /// Default: QuantizerType::NoQuantizer
    pub quantization_type: QuantizerType,

    /// Maximum number of iterations to run for product quantization.
    /// Don't change unless you know what you're doing.
    /// Default: 1000
    pub product_quantization_max_iteration: usize,

    /// Product quantization runs in multiple batches.
    /// Don't change unless you know what you're doing.
    /// Default: 1000
    pub product_quantization_batch_size: usize,

    // Product quantization parameters - dimension of each subvector
    // Default: 8
    pub product_quantization_subvector_dimension: usize,

    // Product quantization parameters - number of bits for each subvector
    // Default: 8
    pub product_quantization_num_bits: usize,

    // Product quantization parameters - number of training rows to build the product quantizer
    // Default: 10000
    pub product_quantization_num_training_rows: usize,

    /// Number of centroids to build. However, the final number of clusters may be more than this
    /// depending on the `max_posting_list_size`, since if a posting list exceeds this size, it
    /// will be split into multiple posting lists.
    /// Default: 10
    pub initial_num_centroids: usize,

    /// Number of data points to to sample for clustering. Often times, we don't want to cluster
    /// all the data points, but instead sample a subset of them. This is useful for large datasets
    /// where we don't want to store all the data points in memory.
    /// Default: 20000
    pub num_data_points_for_clustering: usize,

    /// Number of clusters a vector can belong to. The threshold for adding a vector to a cluster
    /// is controlled by `clustering_distance_threshold_pct`.
    /// Default: 1
    pub max_clusters_per_vector: usize,

    /// If the distance between a vector and a cluster is within this pct of its nearest cluster,
    /// it will also be added to that cluster.
    /// Default: 0.1
    pub clustering_distance_threshold_pct: f32,

    /// Encoding type for posting lists
    /// Default: IntSeqEncodingType::PlainEncoding
    pub posting_list_encoding_type: IntSeqEncodingType,

    /// Specify the size of the memory in bytes for the temporary vector storage
    /// in the builder. If the size exceeds this value, data will be spilled to disk.
    /// Default: 1024 * 1024 * 1024 (1MB)
    pub posting_list_builder_vector_storage_memory_size: usize,

    /// Specify the size of the file in bytes for the temporary vector storage file.
    /// in the builder. If the size exceeds this value, a new temporary file will be created.
    /// Default: 1024 * 1024 * 1024 (1MB)
    pub posting_list_builder_vector_storage_file_size: usize,

    /// Specify max posting list size for clustering centroids.
    /// If the size of a posting list exceeds this value, the posting list will be split into
    /// multiple posting lists.
    /// Default: usize::MAX
    pub max_posting_list_size: usize,

    /// The penalty for unbalanced posting list size in KMeans clustering.
    /// Default: 0.0 (No penalty)
    pub posting_list_kmeans_unbalanced_penalty: f32,

    /// Whether to reindex the collection after building.
    /// This will significantly improve the I/O performance, with the trade-off of
    /// increased build time.
    /// Default: true
    pub reindex: bool,

    /// The size of the WAL file.
    /// Default: 0 (not using WAL)
    #[serde(default = "default_wal_file_size")]
    pub wal_file_size: u64,

    /// Maximum follower entries per WAL write group before closing for batch processing.
    /// Small values = lower latency, higher sync overhead. Large values = higher throughput, higher latency.
    /// Default: 940
    #[serde(default = "default_wal_write_group_size")]
    pub wal_write_group_size: usize,

    /// The maximum number of pending operations before flushing.
    /// Default: 0 (not using max pending ops)
    #[serde(default = "default_max_pending_ops")]
    pub max_pending_ops: u64,

    /// The maximum time to wait before flushing.
    /// Default: 0 (not using max time to flush)
    #[serde(default = "default_max_time_to_flush_ms")]
    pub max_time_to_flush_ms: u64,

    /// Try to maintain the max number of segments.
    /// More segments means slower query as well as deletes.
    #[serde(default = "default_max_number_of_segments")]
    pub max_number_of_segments: usize,

    /// The false positive rate for bloom filter.
    /// Default: 0.01
    #[serde(default = "default_fpr")]
    pub fpr: f64,

    /// Attribute schema for the collection
    pub attribute_schema: Option<AttributeSchema>,
}

fn default_wal_file_size() -> u64 {
    0
}

fn default_wal_write_group_size() -> usize {
    940
}

fn default_max_pending_ops() -> u64 {
    0
}

fn default_max_time_to_flush_ms() -> u64 {
    10000
}

fn default_max_number_of_segments() -> usize {
    10
}

fn default_fpr() -> f64 {
    0.01
}

impl Default for CollectionConfig {
    fn default() -> Self {
        Self {
            centroids_max_neighbors: 10,
            centroids_max_layers: 10,
            centroids_ef_construction: 100,
            centroids_builder_vector_storage_memory_size: 1024 * 1024 * 1024,
            centroids_builder_vector_storage_file_size: 1024 * 1024 * 1024,
            num_features: 768,
            quantization_type: QuantizerType::NoQuantizer,
            product_quantization_max_iteration: 1000,
            product_quantization_batch_size: 1000,
            product_quantization_subvector_dimension: 8,
            product_quantization_num_bits: 8,
            product_quantization_num_training_rows: 10000,
            initial_num_centroids: 10,
            num_data_points_for_clustering: 20000,
            max_clusters_per_vector: 1,
            clustering_distance_threshold_pct: 0.1,
            posting_list_encoding_type: IntSeqEncodingType::PlainEncoding,
            posting_list_builder_vector_storage_memory_size: 1024 * 1024 * 1024,
            posting_list_builder_vector_storage_file_size: 1024 * 1024 * 1024,
            max_posting_list_size: usize::MAX,
            posting_list_kmeans_unbalanced_penalty: 0.0,
            reindex: true,
            wal_file_size: 0,
            wal_write_group_size: default_wal_write_group_size(),
            max_pending_ops: 0,
            max_time_to_flush_ms: 0,
            max_number_of_segments: 10,
            fpr: default_fpr(),
            attribute_schema: None,
        }
    }
}

impl CollectionConfig {
    pub fn default_test_config() -> Self {
        Self {
            num_features: 4,
            centroids_max_neighbors: 10,
            centroids_max_layers: 2,
            centroids_ef_construction: 100,
            centroids_builder_vector_storage_memory_size: 1024,
            centroids_builder_vector_storage_file_size: 1024,
            product_quantization_max_iteration: 1000,
            product_quantization_batch_size: 1,
            product_quantization_subvector_dimension: 2,
            product_quantization_num_bits: 2,
            product_quantization_num_training_rows: 100,
            initial_num_centroids: 10,
            num_data_points_for_clustering: 1000,
            max_clusters_per_vector: 1,
            clustering_distance_threshold_pct: 0.1,
            posting_list_encoding_type: IntSeqEncodingType::EliasFano,
            posting_list_builder_vector_storage_memory_size: 1024,
            posting_list_builder_vector_storage_file_size: 1024,
            max_posting_list_size: usize::MAX,
            posting_list_kmeans_unbalanced_penalty: 0.1,
            reindex: true,
            quantization_type: QuantizerType::NoQuantizer,
            wal_file_size: 1024 * 1024 * 1024,
            wal_write_group_size: default_wal_write_group_size(),
            max_pending_ops: 10000,
            max_time_to_flush_ms: 10000,
            max_number_of_segments: 10,
            fpr: default_fpr(),
            attribute_schema: None,
        }
    }
}
