use prometheus_client::encoding::EncodeLabelSet;
use prometheus_client::metrics::counter::Counter;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::histogram::{Histogram, exponential_buckets};
use prometheus_client::registry::Registry;

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct CollectionRequestLabel {
    pub request_name: String,
    pub collection_name: String,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct RequestLabel {
    pub name: String,
}

pub struct APIMetrics {
    pub num_requests: Family<CollectionRequestLabel, Counter>,
    pub request_latency_ms: Family<RequestLabel, Histogram>,
}

impl Default for APIMetrics {
    fn default() -> Self {
        Self {
            num_requests: Family::<CollectionRequestLabel, Counter>::default(),
            request_latency_ms: Family::<RequestLabel, Histogram>::new_with_constructor(|| {
                Histogram::new(exponential_buckets(1.0, 2.0, 10))
            }),
        }
    }
}

impl APIMetrics {
    pub fn register_metrics(&self, metrics_registry: &mut Registry) {
        metrics_registry.register(
            "num_requests",
            "Number of requests made to a collection",
            self.num_requests.clone(),
        );
        metrics_registry.register(
            "request_latency_ms",
            "Latency for successful requests in milliseconds",
            self.request_latency_ms.clone(),
        );
    }

    pub fn request_latency_ms_observe(&self, request_name: &str, latency_ms: f64) {
        self.request_latency_ms
            .get_or_create(&RequestLabel {
                name: request_name.to_string(),
            })
            .observe(latency_ms);
    }

    pub fn num_requests_inc(&self, request_name: &str, collection_name: &str) {
        self.num_requests
            .get_or_create(&CollectionRequestLabel {
                request_name: request_name.to_string(),
                collection_name: collection_name.to_string(),
            })
            .inc();
    }
}
