use serde::{Deserialize, Serialize};

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
    pub centroids_vector_storage_memory_size: usize,

    /// Specify the size of the file in bytes for the HNSW temporary vector storage file.
    /// in the builder. If the size exceeds this value, a new temporary file will be created.
    /// Default: 1024 * 1024 * 1024 (1MB)
    pub centroids_vector_storage_file_size: usize,

    /// Maximum number of iterations to run for product quantization.
    /// Don't change unless you know what you're doing.
    /// Default: 1000
    pub product_quantization_max_iteration: usize,

    /// Product quantization runs in multiple batches.
    /// Don't change unless you know what you're doing.
    /// Default: 1000
    pub product_quantization_batch_size: usize,

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

    /// Specify the size of the memory in bytes for the temporary vector storage
    /// in the builder. If the size exceeds this value, data will be spilled to disk.
    /// Default: 1024 * 1024 * 1024 (1MB)
    pub builder_vector_storage_memory_size: usize,

    /// Specify the size of the file in bytes for the temporary vector storage file.
    /// in the builder. If the size exceeds this value, a new temporary file will be created.
    /// Default: 1024 * 1024 * 1024 (1MB)
    pub builder_vector_storage_file_size: usize,

    /// Specify max posting list size for clustering centroids.
    /// If the size of a posting list exceeds this value, the posting list will be split into
    /// multiple posting lists.
    /// Default: usize::MAX
    pub max_posting_list_size: usize,

    /// Whether to reindex the collection after building.
    /// This will significantly improve the I/O performance, with the trade-off of
    /// increased build time.
    /// Default: true
    pub reindex: bool,
}

impl Default for CollectionConfig {
    fn default() -> Self {
        Self {
            centroids_max_neighbors: 10,
            centroids_max_layers: 10,
            centroids_ef_construction: 100,
            centroids_vector_storage_memory_size: 1024 * 1024 * 1024,
            centroids_vector_storage_file_size: 1024 * 1024 * 1024,
            num_features: 768,
            product_quantization_max_iteration: 1000,
            product_quantization_batch_size: 1000,
            initial_num_centroids: 10,
            num_data_points_for_clustering: 20000,
            max_clusters_per_vector: 1,
            clustering_distance_threshold_pct: 0.1,
            builder_vector_storage_memory_size: 1024 * 1024 * 1024,
            builder_vector_storage_file_size: 1024 * 1024 * 1024,
            max_posting_list_size: usize::MAX,
            reindex: true,
        }
    }
}
