use prometheus_client::encoding::EncodeLabelSet;
use prometheus_client::metrics::counter::Counter;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::gauge::Gauge;
use prometheus_client::registry::Registry;

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct CollectionLabel {
    pub name: String,
}

#[derive(Default)]
pub struct InternalMetrics {
    pub prometheus_requests: Counter<u64>,
    pub num_collections: Gauge<i64>,
    pub num_active_segments: Family<CollectionLabel, Gauge>,
    pub num_searchable_docs: Family<CollectionLabel, Gauge>,
}

impl InternalMetrics {
    pub fn register_metrics(&self, metrics_registry: &mut Registry) {
        metrics_registry.register(
            "metrics_requests",
            "Number of requests made to the metrics endpoint",
            self.prometheus_requests.clone(),
        );
        metrics_registry.register(
            "num_collections",
            "Number of collections",
            self.num_collections.clone(),
        );
        metrics_registry.register(
            "num_active_segments",
            "Number of active segments in collection",
            self.num_active_segments.clone(),
        );
        // NOTE: (hung) Name this num_documents for now, but it should be changed to num_searchable_docs
        metrics_registry.register(
            "num_documents",
            "Number of documents in collection",
            self.num_searchable_docs.clone(),
        );
    }

    pub fn num_searchable_docs_set(&self, collection_name: &str, num_docs: i64) {
        let label = CollectionLabel {
            name: collection_name.to_string(),
        };
        self.num_searchable_docs.get_or_create(&label).set(num_docs);
    }

    pub fn num_searchable_docs_dec(&self, collection_name: &str) {
        let label = CollectionLabel {
            name: collection_name.to_string(),
        };
        self.num_searchable_docs.get_or_create(&label).dec();
    }

    pub fn num_searchable_docs_dec_by(&self, collection_name: &str, num_docs: i64) {
        let label = CollectionLabel {
            name: collection_name.to_string(),
        };
        self.num_searchable_docs
            .get_or_create(&label)
            .dec_by(num_docs);
    }

    pub fn num_searchable_docs_inc(&self, collection_name: &str) {
        let label = CollectionLabel {
            name: collection_name.to_string(),
        };
        self.num_searchable_docs.get_or_create(&label).inc();
    }

    pub fn num_searchable_docs_inc_by(&self, collection_name: &str, num_docs: i64) {
        let label = CollectionLabel {
            name: collection_name.to_string(),
        };
        self.num_searchable_docs
            .get_or_create(&label)
            .inc_by(num_docs);
    }

    pub fn num_searchable_docs_get(&self, collection_name: &str) -> i64 {
        let label = CollectionLabel {
            name: collection_name.to_string(),
        };
        self.num_searchable_docs
            .get(&label)
            .map(|metric| metric.get())
            .unwrap_or(0)
    }

    pub fn num_active_segments_get(&self, collection_name: &str) -> i64 {
        let label = CollectionLabel {
            name: collection_name.to_string(),
        };
        self.num_active_segments
            .get(&label)
            .map(|metric| metric.get())
            .unwrap_or(0)
    }

    pub fn num_active_segments_set(&self, collection_name: &str, num_segments: i64) {
        let label = CollectionLabel {
            name: collection_name.to_string(),
        };
        self.num_active_segments
            .get_or_create(&label)
            .set(num_segments);
    }
}
