use lazy_static::lazy_static;
use prometheus_client::encoding::EncodeLabelSet;
use prometheus_client::metrics::counter::Counter;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::gauge::Gauge;
use prometheus_client::registry::Registry;

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct CollectionLabel {
    pub name: String,
}

lazy_static! {
    /// Add your own metrics here. Remember to add them to the `register_metrics` function.
    /// Example: a counter metric for the number of incoming requests to the metrics endpoint.
    pub static ref METRICS_REQUESTS: Counter<u64> = Counter::default();
    pub static ref NUM_COLLECTIONS: Gauge<i64> = Gauge::default();
    pub static ref NUM_ACTIVE_SEGMENTS: Family<CollectionLabel, Gauge> = Family::<CollectionLabel, Gauge>::default();
    pub static ref NUM_SEARCHABLE_DOCS: Family<CollectionLabel, Gauge> = Family::<CollectionLabel, Gauge>::default();
}

/// Register the metrics with the provided registry.
pub fn register_metrics(metrics_registry: &mut Registry) {
    metrics_registry.register(
        "metrics_requests",
        "Number of requests made to the metrics endpoint",
        METRICS_REQUESTS.clone(),
    );
    metrics_registry.register(
        "num_collections",
        "Number of collections",
        NUM_COLLECTIONS.clone(),
    );
    metrics_registry.register(
        "num_active_segments_per_collection",
        "Number of active segments per collection",
        NUM_ACTIVE_SEGMENTS.clone(),
    );
    metrics_registry.register(
        "num_searchable_docs_per_collection",
        "Number of searchable documents per collection",
        NUM_SEARCHABLE_DOCS.clone(),
    );
}
