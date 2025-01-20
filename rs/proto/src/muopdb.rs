#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GetRequest {
    #[prost(string, tag = "1")]
    pub index: ::prost::alloc::string::String,
    #[prost(float, repeated, tag = "2")]
    pub vector: ::prost::alloc::vec::Vec<f32>,
    #[prost(uint32, tag = "3")]
    pub top_k: u32,
    #[prost(uint32, tag = "5")]
    pub ef_construction: u32,
    /// For metrics, don't set by default
    #[prost(bool, tag = "4")]
    pub record_metrics: bool,
    #[prost(uint64, repeated, tag = "6")]
    pub low_user_ids: ::prost::alloc::vec::Vec<u64>,
    #[prost(uint64, repeated, tag = "7")]
    pub high_user_ids: ::prost::alloc::vec::Vec<u64>,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GetResponse {
    #[prost(uint64, repeated, tag = "1")]
    pub low_ids: ::prost::alloc::vec::Vec<u64>,
    #[prost(uint64, repeated, tag = "3")]
    pub high_ids: ::prost::alloc::vec::Vec<u64>,
    /// For metrics, not enabled by default
    #[prost(uint64, tag = "2")]
    pub num_pages_accessed: u64,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GetSegmentsRequest {
    #[prost(string, tag = "1")]
    pub collection_name: ::prost::alloc::string::String,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GetSegmentsResponse {
    #[prost(string, repeated, tag = "1")]
    pub segment_names: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CompactSegmentsRequest {
    #[prost(string, tag = "1")]
    pub collection_name: ::prost::alloc::string::String,
    #[prost(string, repeated, tag = "2")]
    pub segment_names: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CompactSegmentsResponse {}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CreateCollectionRequest {
    #[prost(string, tag = "1")]
    pub collection_name: ::prost::alloc::string::String,
    /// Collection configuration parameters. The default values for these are defined
    /// in `rs/config/src/collection.rs`. You just need to override those that differ
    /// from the default values.
    #[prost(uint32, optional, tag = "3")]
    pub num_features: ::core::option::Option<u32>,
    #[prost(uint32, optional, tag = "4")]
    pub centroids_max_neighbors: ::core::option::Option<u32>,
    #[prost(uint32, optional, tag = "5")]
    pub centroids_max_layers: ::core::option::Option<u32>,
    #[prost(uint32, optional, tag = "6")]
    pub centroids_ef_construction: ::core::option::Option<u32>,
    #[prost(uint64, optional, tag = "7")]
    pub centroids_builder_vector_storage_memory_size: ::core::option::Option<u64>,
    #[prost(uint64, optional, tag = "8")]
    pub centroids_builder_vector_storage_file_size: ::core::option::Option<u64>,
    #[prost(enumeration = "QuantizerType", optional, tag = "9")]
    pub quantization_type: ::core::option::Option<i32>,
    #[prost(uint32, optional, tag = "10")]
    pub product_quantization_max_iteration: ::core::option::Option<u32>,
    #[prost(uint32, optional, tag = "11")]
    pub product_quantization_batch_size: ::core::option::Option<u32>,
    #[prost(uint32, optional, tag = "12")]
    pub product_quantization_subvector_dimension: ::core::option::Option<u32>,
    #[prost(uint32, optional, tag = "13")]
    pub product_quantization_num_bits: ::core::option::Option<u32>,
    #[prost(uint32, optional, tag = "14")]
    pub product_quantization_num_training_rows: ::core::option::Option<u32>,
    #[prost(uint32, optional, tag = "15")]
    pub initial_num_centroids: ::core::option::Option<u32>,
    #[prost(uint32, optional, tag = "16")]
    pub num_data_points_for_clustering: ::core::option::Option<u32>,
    #[prost(uint32, optional, tag = "17")]
    pub max_clusters_per_vector: ::core::option::Option<u32>,
    #[prost(float, optional, tag = "18")]
    pub clustering_distance_threshold_pct: ::core::option::Option<f32>,
    #[prost(enumeration = "IntSeqEncodingType", optional, tag = "19")]
    pub posting_list_encoding_type: ::core::option::Option<i32>,
    #[prost(uint64, optional, tag = "20")]
    pub posting_list_builder_vector_storage_memory_size: ::core::option::Option<u64>,
    #[prost(uint64, optional, tag = "21")]
    pub posting_list_builder_vector_storage_file_size: ::core::option::Option<u64>,
    #[prost(uint64, optional, tag = "22")]
    pub max_posting_list_size: ::core::option::Option<u64>,
    #[prost(float, optional, tag = "23")]
    pub posting_list_kmeans_unbalanced_penalty: ::core::option::Option<f32>,
    #[prost(bool, optional, tag = "24")]
    pub reindex: ::core::option::Option<bool>,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CreateCollectionResponse {}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SearchRequest {
    #[prost(string, tag = "1")]
    pub collection_name: ::prost::alloc::string::String,
    #[prost(float, repeated, tag = "2")]
    pub vector: ::prost::alloc::vec::Vec<f32>,
    #[prost(uint32, tag = "3")]
    pub top_k: u32,
    #[prost(uint32, tag = "5")]
    pub ef_construction: u32,
    /// For metrics, don't set by default.
    /// This has some performance impact on the query.
    #[prost(bool, tag = "4")]
    pub record_metrics: bool,
    /// List of lower 64 bits of the user ids to search for.
    #[prost(uint64, repeated, tag = "6")]
    pub low_user_ids: ::prost::alloc::vec::Vec<u64>,
    /// List of higher 64 bits of the user ids to search for.
    #[prost(uint64, repeated, tag = "7")]
    pub high_user_ids: ::prost::alloc::vec::Vec<u64>,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SearchResponse {
    /// List of lower 64 bits of the doc_ids
    #[prost(uint64, repeated, tag = "1")]
    pub low_ids: ::prost::alloc::vec::Vec<u64>,
    /// List of higher 64 bits of the doc_ids
    #[prost(uint64, repeated, tag = "4")]
    pub high_ids: ::prost::alloc::vec::Vec<u64>,
    #[prost(float, repeated, tag = "2")]
    pub scores: ::prost::alloc::vec::Vec<f32>,
    /// For metrics, not enabled by default
    #[prost(uint64, tag = "3")]
    pub num_pages_accessed: u64,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct InsertRequest {
    #[prost(string, tag = "1")]
    pub collection_name: ::prost::alloc::string::String,
    /// List of lower 64 bits of the doc_ids
    #[prost(uint64, repeated, tag = "2")]
    pub low_ids: ::prost::alloc::vec::Vec<u64>,
    /// List of higher 64 bits of the doc_ids
    #[prost(uint64, repeated, tag = "5")]
    pub high_ids: ::prost::alloc::vec::Vec<u64>,
    /// Flattened vector. If the dimension is 10,
    /// and the number of vectors is 5,
    /// then this list should have 50 elements.
    #[prost(float, repeated, tag = "3")]
    pub vectors: ::prost::alloc::vec::Vec<f32>,
    /// List of lower 64 bits of the user ids.
    #[prost(uint64, repeated, tag = "4")]
    pub low_user_ids: ::prost::alloc::vec::Vec<u64>,
    /// List of higher 64 bits of the user ids.
    #[prost(uint64, repeated, tag = "6")]
    pub high_user_ids: ::prost::alloc::vec::Vec<u64>,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct InsertResponse {
    /// List of lower 64 bits of the doc_ids
    #[prost(uint64, repeated, tag = "1")]
    pub inserted_low_ids: ::prost::alloc::vec::Vec<u64>,
    /// List of higher 64 bits of the doc_ids
    #[prost(uint64, repeated, tag = "2")]
    pub inserted_high_ids: ::prost::alloc::vec::Vec<u64>,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct FlushRequest {
    #[prost(string, tag = "1")]
    pub collection_name: ::prost::alloc::string::String,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct FlushResponse {
    #[prost(string, repeated, tag = "1")]
    pub flushed_segments: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
}
/// If you send a large number of vectors, you can use this method to
/// save some time on serialization and deserialization.
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct InsertPackedRequest {
    #[prost(string, tag = "1")]
    pub collection_name: ::prost::alloc::string::String,
    /// List of lower 64 bits of the doc_ids.
    /// These ids should be packed with 8 bytes per number.
    #[prost(bytes = "vec", tag = "2")]
    pub low_ids: ::prost::alloc::vec::Vec<u8>,
    /// List of lower 64 bits of the doc_ids.
    /// These ids should be packed with 8 bytes per number.
    #[prost(bytes = "vec", tag = "5")]
    pub high_ids: ::prost::alloc::vec::Vec<u8>,
    /// Packed flattened vector. If the dimension is 10,
    /// and the number of vectors is 5, then this list should
    /// have 50 elements, packed with 8 bytes per number.
    #[prost(bytes = "vec", tag = "3")]
    pub vectors: ::prost::alloc::vec::Vec<u8>,
    #[prost(uint64, repeated, tag = "4")]
    pub low_user_ids: ::prost::alloc::vec::Vec<u64>,
    #[prost(uint64, repeated, tag = "6")]
    pub high_user_ids: ::prost::alloc::vec::Vec<u64>,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct InsertPackedResponse {}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum QuantizerType {
    NoQuantizer = 0,
    ProductQuantizer = 1,
}
impl QuantizerType {
    /// String value of the enum field names used in the ProtoBuf definition.
    ///
    /// The values are not transformed in any way and thus are considered stable
    /// (if the ProtoBuf definition does not change) and safe for programmatic use.
    pub fn as_str_name(&self) -> &'static str {
        match self {
            QuantizerType::NoQuantizer => "NO_QUANTIZER",
            QuantizerType::ProductQuantizer => "PRODUCT_QUANTIZER",
        }
    }
    /// Creates an enum from field names used in the ProtoBuf definition.
    pub fn from_str_name(value: &str) -> ::core::option::Option<Self> {
        match value {
            "NO_QUANTIZER" => Some(Self::NoQuantizer),
            "PRODUCT_QUANTIZER" => Some(Self::ProductQuantizer),
            _ => None,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum IntSeqEncodingType {
    PlainEncoding = 0,
    EliasFano = 1,
}
impl IntSeqEncodingType {
    /// String value of the enum field names used in the ProtoBuf definition.
    ///
    /// The values are not transformed in any way and thus are considered stable
    /// (if the ProtoBuf definition does not change) and safe for programmatic use.
    pub fn as_str_name(&self) -> &'static str {
        match self {
            IntSeqEncodingType::PlainEncoding => "PLAIN_ENCODING",
            IntSeqEncodingType::EliasFano => "ELIAS_FANO",
        }
    }
    /// Creates an enum from field names used in the ProtoBuf definition.
    pub fn from_str_name(value: &str) -> ::core::option::Option<Self> {
        match value {
            "PLAIN_ENCODING" => Some(Self::PlainEncoding),
            "ELIAS_FANO" => Some(Self::EliasFano),
            _ => None,
        }
    }
}
/// Generated client implementations.
pub mod aggregator_client {
    #![allow(unused_variables, dead_code, missing_docs, clippy::let_unit_value)]
    use tonic::codegen::http::Uri;
    use tonic::codegen::*;
    #[derive(Debug, Clone)]
    pub struct AggregatorClient<T> {
        inner: tonic::client::Grpc<T>,
    }
    impl AggregatorClient<tonic::transport::Channel> {
        /// Attempt to create a new client by connecting to a given endpoint.
        pub async fn connect<D>(dst: D) -> Result<Self, tonic::transport::Error>
        where
            D: std::convert::TryInto<tonic::transport::Endpoint>,
            D::Error: Into<StdError>,
        {
            let conn = tonic::transport::Endpoint::new(dst)?.connect().await?;
            Ok(Self::new(conn))
        }
    }
    impl<T> AggregatorClient<T>
    where
        T: tonic::client::GrpcService<tonic::body::BoxBody>,
        T::Error: Into<StdError>,
        T::ResponseBody: Body<Data = Bytes> + Send + 'static,
        <T::ResponseBody as Body>::Error: Into<StdError> + Send,
    {
        pub fn new(inner: T) -> Self {
            let inner = tonic::client::Grpc::new(inner);
            Self { inner }
        }
        pub fn with_origin(inner: T, origin: Uri) -> Self {
            let inner = tonic::client::Grpc::with_origin(inner, origin);
            Self { inner }
        }
        pub fn with_interceptor<F>(
            inner: T,
            interceptor: F,
        ) -> AggregatorClient<InterceptedService<T, F>>
        where
            F: tonic::service::Interceptor,
            T::ResponseBody: Default,
            T: tonic::codegen::Service<
                http::Request<tonic::body::BoxBody>,
                Response = http::Response<
                    <T as tonic::client::GrpcService<tonic::body::BoxBody>>::ResponseBody,
                >,
            >,
            <T as tonic::codegen::Service<http::Request<tonic::body::BoxBody>>>::Error:
                Into<StdError> + Send + Sync,
        {
            AggregatorClient::new(InterceptedService::new(inner, interceptor))
        }
        /// Compress requests with the given encoding.
        ///
        /// This requires the server to support it otherwise it might respond with an
        /// error.
        #[must_use]
        pub fn send_compressed(mut self, encoding: CompressionEncoding) -> Self {
            self.inner = self.inner.send_compressed(encoding);
            self
        }
        /// Enable decompressing responses.
        #[must_use]
        pub fn accept_compressed(mut self, encoding: CompressionEncoding) -> Self {
            self.inner = self.inner.accept_compressed(encoding);
            self
        }
        pub async fn get(
            &mut self,
            request: impl tonic::IntoRequest<super::GetRequest>,
        ) -> Result<tonic::Response<super::GetResponse>, tonic::Status> {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::new(
                    tonic::Code::Unknown,
                    format!("Service was not ready: {}", e.into()),
                )
            })?;
            let codec = tonic::codec::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static("/muopdb.Aggregator/Get");
            self.inner.unary(request.into_request(), path, codec).await
        }
    }
}
/// Generated client implementations.
pub mod index_server_client {
    #![allow(unused_variables, dead_code, missing_docs, clippy::let_unit_value)]
    use tonic::codegen::http::Uri;
    use tonic::codegen::*;
    #[derive(Debug, Clone)]
    pub struct IndexServerClient<T> {
        inner: tonic::client::Grpc<T>,
    }
    impl IndexServerClient<tonic::transport::Channel> {
        /// Attempt to create a new client by connecting to a given endpoint.
        pub async fn connect<D>(dst: D) -> Result<Self, tonic::transport::Error>
        where
            D: std::convert::TryInto<tonic::transport::Endpoint>,
            D::Error: Into<StdError>,
        {
            let conn = tonic::transport::Endpoint::new(dst)?.connect().await?;
            Ok(Self::new(conn))
        }
    }
    impl<T> IndexServerClient<T>
    where
        T: tonic::client::GrpcService<tonic::body::BoxBody>,
        T::Error: Into<StdError>,
        T::ResponseBody: Body<Data = Bytes> + Send + 'static,
        <T::ResponseBody as Body>::Error: Into<StdError> + Send,
    {
        pub fn new(inner: T) -> Self {
            let inner = tonic::client::Grpc::new(inner);
            Self { inner }
        }
        pub fn with_origin(inner: T, origin: Uri) -> Self {
            let inner = tonic::client::Grpc::with_origin(inner, origin);
            Self { inner }
        }
        pub fn with_interceptor<F>(
            inner: T,
            interceptor: F,
        ) -> IndexServerClient<InterceptedService<T, F>>
        where
            F: tonic::service::Interceptor,
            T::ResponseBody: Default,
            T: tonic::codegen::Service<
                http::Request<tonic::body::BoxBody>,
                Response = http::Response<
                    <T as tonic::client::GrpcService<tonic::body::BoxBody>>::ResponseBody,
                >,
            >,
            <T as tonic::codegen::Service<http::Request<tonic::body::BoxBody>>>::Error:
                Into<StdError> + Send + Sync,
        {
            IndexServerClient::new(InterceptedService::new(inner, interceptor))
        }
        /// Compress requests with the given encoding.
        ///
        /// This requires the server to support it otherwise it might respond with an
        /// error.
        #[must_use]
        pub fn send_compressed(mut self, encoding: CompressionEncoding) -> Self {
            self.inner = self.inner.send_compressed(encoding);
            self
        }
        /// Enable decompressing responses.
        #[must_use]
        pub fn accept_compressed(mut self, encoding: CompressionEncoding) -> Self {
            self.inner = self.inner.accept_compressed(encoding);
            self
        }
        pub async fn create_collection(
            &mut self,
            request: impl tonic::IntoRequest<super::CreateCollectionRequest>,
        ) -> Result<tonic::Response<super::CreateCollectionResponse>, tonic::Status> {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::new(
                    tonic::Code::Unknown,
                    format!("Service was not ready: {}", e.into()),
                )
            })?;
            let codec = tonic::codec::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static("/muopdb.IndexServer/CreateCollection");
            self.inner.unary(request.into_request(), path, codec).await
        }
        pub async fn search(
            &mut self,
            request: impl tonic::IntoRequest<super::SearchRequest>,
        ) -> Result<tonic::Response<super::SearchResponse>, tonic::Status> {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::new(
                    tonic::Code::Unknown,
                    format!("Service was not ready: {}", e.into()),
                )
            })?;
            let codec = tonic::codec::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static("/muopdb.IndexServer/Search");
            self.inner.unary(request.into_request(), path, codec).await
        }
        pub async fn insert(
            &mut self,
            request: impl tonic::IntoRequest<super::InsertRequest>,
        ) -> Result<tonic::Response<super::InsertResponse>, tonic::Status> {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::new(
                    tonic::Code::Unknown,
                    format!("Service was not ready: {}", e.into()),
                )
            })?;
            let codec = tonic::codec::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static("/muopdb.IndexServer/Insert");
            self.inner.unary(request.into_request(), path, codec).await
        }
        pub async fn insert_packed(
            &mut self,
            request: impl tonic::IntoRequest<super::InsertPackedRequest>,
        ) -> Result<tonic::Response<super::InsertPackedResponse>, tonic::Status> {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::new(
                    tonic::Code::Unknown,
                    format!("Service was not ready: {}", e.into()),
                )
            })?;
            let codec = tonic::codec::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static("/muopdb.IndexServer/InsertPacked");
            self.inner.unary(request.into_request(), path, codec).await
        }
        pub async fn flush(
            &mut self,
            request: impl tonic::IntoRequest<super::FlushRequest>,
        ) -> Result<tonic::Response<super::FlushResponse>, tonic::Status> {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::new(
                    tonic::Code::Unknown,
                    format!("Service was not ready: {}", e.into()),
                )
            })?;
            let codec = tonic::codec::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static("/muopdb.IndexServer/Flush");
            self.inner.unary(request.into_request(), path, codec).await
        }
        pub async fn get_segments(
            &mut self,
            request: impl tonic::IntoRequest<super::GetSegmentsRequest>,
        ) -> Result<tonic::Response<super::GetSegmentsResponse>, tonic::Status> {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::new(
                    tonic::Code::Unknown,
                    format!("Service was not ready: {}", e.into()),
                )
            })?;
            let codec = tonic::codec::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static("/muopdb.IndexServer/GetSegments");
            self.inner.unary(request.into_request(), path, codec).await
        }
        pub async fn compact_segments(
            &mut self,
            request: impl tonic::IntoRequest<super::CompactSegmentsRequest>,
        ) -> Result<tonic::Response<super::CompactSegmentsResponse>, tonic::Status> {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::new(
                    tonic::Code::Unknown,
                    format!("Service was not ready: {}", e.into()),
                )
            })?;
            let codec = tonic::codec::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static("/muopdb.IndexServer/CompactSegments");
            self.inner.unary(request.into_request(), path, codec).await
        }
    }
}
/// Generated server implementations.
pub mod aggregator_server {
    #![allow(unused_variables, dead_code, missing_docs, clippy::let_unit_value)]
    use tonic::codegen::*;
    /// Generated trait containing gRPC methods that should be implemented for use with AggregatorServer.
    #[async_trait]
    pub trait Aggregator: Send + Sync + 'static {
        async fn get(
            &self,
            request: tonic::Request<super::GetRequest>,
        ) -> Result<tonic::Response<super::GetResponse>, tonic::Status>;
    }
    #[derive(Debug)]
    pub struct AggregatorServer<T: Aggregator> {
        inner: _Inner<T>,
        accept_compression_encodings: EnabledCompressionEncodings,
        send_compression_encodings: EnabledCompressionEncodings,
    }
    struct _Inner<T>(Arc<T>);
    impl<T: Aggregator> AggregatorServer<T> {
        pub fn new(inner: T) -> Self {
            Self::from_arc(Arc::new(inner))
        }
        pub fn from_arc(inner: Arc<T>) -> Self {
            let inner = _Inner(inner);
            Self {
                inner,
                accept_compression_encodings: Default::default(),
                send_compression_encodings: Default::default(),
            }
        }
        pub fn with_interceptor<F>(inner: T, interceptor: F) -> InterceptedService<Self, F>
        where
            F: tonic::service::Interceptor,
        {
            InterceptedService::new(Self::new(inner), interceptor)
        }
        /// Enable decompressing requests with the given encoding.
        #[must_use]
        pub fn accept_compressed(mut self, encoding: CompressionEncoding) -> Self {
            self.accept_compression_encodings.enable(encoding);
            self
        }
        /// Compress responses with the given encoding, if the client supports it.
        #[must_use]
        pub fn send_compressed(mut self, encoding: CompressionEncoding) -> Self {
            self.send_compression_encodings.enable(encoding);
            self
        }
    }
    impl<T, B> tonic::codegen::Service<http::Request<B>> for AggregatorServer<T>
    where
        T: Aggregator,
        B: Body + Send + 'static,
        B::Error: Into<StdError> + Send + 'static,
    {
        type Response = http::Response<tonic::body::BoxBody>;
        type Error = std::convert::Infallible;
        type Future = BoxFuture<Self::Response, Self::Error>;
        fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
            Poll::Ready(Ok(()))
        }
        fn call(&mut self, req: http::Request<B>) -> Self::Future {
            let inner = self.inner.clone();
            match req.uri().path() {
                "/muopdb.Aggregator/Get" => {
                    #[allow(non_camel_case_types)]
                    struct GetSvc<T: Aggregator>(pub Arc<T>);
                    impl<T: Aggregator> tonic::server::UnaryService<super::GetRequest> for GetSvc<T> {
                        type Response = super::GetResponse;
                        type Future = BoxFuture<tonic::Response<Self::Response>, tonic::Status>;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::GetRequest>,
                        ) -> Self::Future {
                            let inner = self.0.clone();
                            let fut = async move { (*inner).get(request).await };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let inner = inner.0;
                        let method = GetSvc(inner);
                        let codec = tonic::codec::ProstCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec).apply_compression_config(
                            accept_compression_encodings,
                            send_compression_encodings,
                        );
                        let res = grpc.unary(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                _ => Box::pin(async move {
                    Ok(http::Response::builder()
                        .status(200)
                        .header("grpc-status", "12")
                        .header("content-type", "application/grpc")
                        .body(empty_body())
                        .unwrap())
                }),
            }
        }
    }
    impl<T: Aggregator> Clone for AggregatorServer<T> {
        fn clone(&self) -> Self {
            let inner = self.inner.clone();
            Self {
                inner,
                accept_compression_encodings: self.accept_compression_encodings,
                send_compression_encodings: self.send_compression_encodings,
            }
        }
    }
    impl<T: Aggregator> Clone for _Inner<T> {
        fn clone(&self) -> Self {
            Self(self.0.clone())
        }
    }
    impl<T: std::fmt::Debug> std::fmt::Debug for _Inner<T> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{:?}", self.0)
        }
    }
    impl<T: Aggregator> tonic::server::NamedService for AggregatorServer<T> {
        const NAME: &'static str = "muopdb.Aggregator";
    }
}
/// Generated server implementations.
pub mod index_server_server {
    #![allow(unused_variables, dead_code, missing_docs, clippy::let_unit_value)]
    use tonic::codegen::*;
    /// Generated trait containing gRPC methods that should be implemented for use with IndexServerServer.
    #[async_trait]
    pub trait IndexServer: Send + Sync + 'static {
        async fn create_collection(
            &self,
            request: tonic::Request<super::CreateCollectionRequest>,
        ) -> Result<tonic::Response<super::CreateCollectionResponse>, tonic::Status>;
        async fn search(
            &self,
            request: tonic::Request<super::SearchRequest>,
        ) -> Result<tonic::Response<super::SearchResponse>, tonic::Status>;
        async fn insert(
            &self,
            request: tonic::Request<super::InsertRequest>,
        ) -> Result<tonic::Response<super::InsertResponse>, tonic::Status>;
        async fn insert_packed(
            &self,
            request: tonic::Request<super::InsertPackedRequest>,
        ) -> Result<tonic::Response<super::InsertPackedResponse>, tonic::Status>;
        async fn flush(
            &self,
            request: tonic::Request<super::FlushRequest>,
        ) -> Result<tonic::Response<super::FlushResponse>, tonic::Status>;
        async fn get_segments(
            &self,
            request: tonic::Request<super::GetSegmentsRequest>,
        ) -> Result<tonic::Response<super::GetSegmentsResponse>, tonic::Status>;
        async fn compact_segments(
            &self,
            request: tonic::Request<super::CompactSegmentsRequest>,
        ) -> Result<tonic::Response<super::CompactSegmentsResponse>, tonic::Status>;
    }
    #[derive(Debug)]
    pub struct IndexServerServer<T: IndexServer> {
        inner: _Inner<T>,
        accept_compression_encodings: EnabledCompressionEncodings,
        send_compression_encodings: EnabledCompressionEncodings,
    }
    struct _Inner<T>(Arc<T>);
    impl<T: IndexServer> IndexServerServer<T> {
        pub fn new(inner: T) -> Self {
            Self::from_arc(Arc::new(inner))
        }
        pub fn from_arc(inner: Arc<T>) -> Self {
            let inner = _Inner(inner);
            Self {
                inner,
                accept_compression_encodings: Default::default(),
                send_compression_encodings: Default::default(),
            }
        }
        pub fn with_interceptor<F>(inner: T, interceptor: F) -> InterceptedService<Self, F>
        where
            F: tonic::service::Interceptor,
        {
            InterceptedService::new(Self::new(inner), interceptor)
        }
        /// Enable decompressing requests with the given encoding.
        #[must_use]
        pub fn accept_compressed(mut self, encoding: CompressionEncoding) -> Self {
            self.accept_compression_encodings.enable(encoding);
            self
        }
        /// Compress responses with the given encoding, if the client supports it.
        #[must_use]
        pub fn send_compressed(mut self, encoding: CompressionEncoding) -> Self {
            self.send_compression_encodings.enable(encoding);
            self
        }
    }
    impl<T, B> tonic::codegen::Service<http::Request<B>> for IndexServerServer<T>
    where
        T: IndexServer,
        B: Body + Send + 'static,
        B::Error: Into<StdError> + Send + 'static,
    {
        type Response = http::Response<tonic::body::BoxBody>;
        type Error = std::convert::Infallible;
        type Future = BoxFuture<Self::Response, Self::Error>;
        fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
            Poll::Ready(Ok(()))
        }
        fn call(&mut self, req: http::Request<B>) -> Self::Future {
            let inner = self.inner.clone();
            match req.uri().path() {
                "/muopdb.IndexServer/CreateCollection" => {
                    #[allow(non_camel_case_types)]
                    struct CreateCollectionSvc<T: IndexServer>(pub Arc<T>);
                    impl<T: IndexServer> tonic::server::UnaryService<super::CreateCollectionRequest>
                        for CreateCollectionSvc<T>
                    {
                        type Response = super::CreateCollectionResponse;
                        type Future = BoxFuture<tonic::Response<Self::Response>, tonic::Status>;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::CreateCollectionRequest>,
                        ) -> Self::Future {
                            let inner = self.0.clone();
                            let fut = async move { (*inner).create_collection(request).await };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let inner = inner.0;
                        let method = CreateCollectionSvc(inner);
                        let codec = tonic::codec::ProstCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec).apply_compression_config(
                            accept_compression_encodings,
                            send_compression_encodings,
                        );
                        let res = grpc.unary(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                "/muopdb.IndexServer/Search" => {
                    #[allow(non_camel_case_types)]
                    struct SearchSvc<T: IndexServer>(pub Arc<T>);
                    impl<T: IndexServer> tonic::server::UnaryService<super::SearchRequest> for SearchSvc<T> {
                        type Response = super::SearchResponse;
                        type Future = BoxFuture<tonic::Response<Self::Response>, tonic::Status>;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::SearchRequest>,
                        ) -> Self::Future {
                            let inner = self.0.clone();
                            let fut = async move { (*inner).search(request).await };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let inner = inner.0;
                        let method = SearchSvc(inner);
                        let codec = tonic::codec::ProstCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec).apply_compression_config(
                            accept_compression_encodings,
                            send_compression_encodings,
                        );
                        let res = grpc.unary(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                "/muopdb.IndexServer/Insert" => {
                    #[allow(non_camel_case_types)]
                    struct InsertSvc<T: IndexServer>(pub Arc<T>);
                    impl<T: IndexServer> tonic::server::UnaryService<super::InsertRequest> for InsertSvc<T> {
                        type Response = super::InsertResponse;
                        type Future = BoxFuture<tonic::Response<Self::Response>, tonic::Status>;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::InsertRequest>,
                        ) -> Self::Future {
                            let inner = self.0.clone();
                            let fut = async move { (*inner).insert(request).await };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let inner = inner.0;
                        let method = InsertSvc(inner);
                        let codec = tonic::codec::ProstCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec).apply_compression_config(
                            accept_compression_encodings,
                            send_compression_encodings,
                        );
                        let res = grpc.unary(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                "/muopdb.IndexServer/InsertPacked" => {
                    #[allow(non_camel_case_types)]
                    struct InsertPackedSvc<T: IndexServer>(pub Arc<T>);
                    impl<T: IndexServer> tonic::server::UnaryService<super::InsertPackedRequest>
                        for InsertPackedSvc<T>
                    {
                        type Response = super::InsertPackedResponse;
                        type Future = BoxFuture<tonic::Response<Self::Response>, tonic::Status>;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::InsertPackedRequest>,
                        ) -> Self::Future {
                            let inner = self.0.clone();
                            let fut = async move { (*inner).insert_packed(request).await };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let inner = inner.0;
                        let method = InsertPackedSvc(inner);
                        let codec = tonic::codec::ProstCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec).apply_compression_config(
                            accept_compression_encodings,
                            send_compression_encodings,
                        );
                        let res = grpc.unary(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                "/muopdb.IndexServer/Flush" => {
                    #[allow(non_camel_case_types)]
                    struct FlushSvc<T: IndexServer>(pub Arc<T>);
                    impl<T: IndexServer> tonic::server::UnaryService<super::FlushRequest> for FlushSvc<T> {
                        type Response = super::FlushResponse;
                        type Future = BoxFuture<tonic::Response<Self::Response>, tonic::Status>;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::FlushRequest>,
                        ) -> Self::Future {
                            let inner = self.0.clone();
                            let fut = async move { (*inner).flush(request).await };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let inner = inner.0;
                        let method = FlushSvc(inner);
                        let codec = tonic::codec::ProstCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec).apply_compression_config(
                            accept_compression_encodings,
                            send_compression_encodings,
                        );
                        let res = grpc.unary(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                "/muopdb.IndexServer/GetSegments" => {
                    #[allow(non_camel_case_types)]
                    struct GetSegmentsSvc<T: IndexServer>(pub Arc<T>);
                    impl<T: IndexServer> tonic::server::UnaryService<super::GetSegmentsRequest> for GetSegmentsSvc<T> {
                        type Response = super::GetSegmentsResponse;
                        type Future = BoxFuture<tonic::Response<Self::Response>, tonic::Status>;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::GetSegmentsRequest>,
                        ) -> Self::Future {
                            let inner = self.0.clone();
                            let fut = async move { (*inner).get_segments(request).await };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let inner = inner.0;
                        let method = GetSegmentsSvc(inner);
                        let codec = tonic::codec::ProstCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec).apply_compression_config(
                            accept_compression_encodings,
                            send_compression_encodings,
                        );
                        let res = grpc.unary(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                "/muopdb.IndexServer/CompactSegments" => {
                    #[allow(non_camel_case_types)]
                    struct CompactSegmentsSvc<T: IndexServer>(pub Arc<T>);
                    impl<T: IndexServer> tonic::server::UnaryService<super::CompactSegmentsRequest>
                        for CompactSegmentsSvc<T>
                    {
                        type Response = super::CompactSegmentsResponse;
                        type Future = BoxFuture<tonic::Response<Self::Response>, tonic::Status>;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::CompactSegmentsRequest>,
                        ) -> Self::Future {
                            let inner = self.0.clone();
                            let fut = async move { (*inner).compact_segments(request).await };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let inner = inner.0;
                        let method = CompactSegmentsSvc(inner);
                        let codec = tonic::codec::ProstCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec).apply_compression_config(
                            accept_compression_encodings,
                            send_compression_encodings,
                        );
                        let res = grpc.unary(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                _ => Box::pin(async move {
                    Ok(http::Response::builder()
                        .status(200)
                        .header("grpc-status", "12")
                        .header("content-type", "application/grpc")
                        .body(empty_body())
                        .unwrap())
                }),
            }
        }
    }
    impl<T: IndexServer> Clone for IndexServerServer<T> {
        fn clone(&self) -> Self {
            let inner = self.inner.clone();
            Self {
                inner,
                accept_compression_encodings: self.accept_compression_encodings,
                send_compression_encodings: self.send_compression_encodings,
            }
        }
    }
    impl<T: IndexServer> Clone for _Inner<T> {
        fn clone(&self) -> Self {
            Self(self.0.clone())
        }
    }
    impl<T: std::fmt::Debug> std::fmt::Debug for _Inner<T> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{:?}", self.0)
        }
    }
    impl<T: IndexServer> tonic::server::NamedService for IndexServerServer<T> {
        const NAME: &'static str = "muopdb.IndexServer";
    }
}
