mod api;
mod internal;

use api::APIMetrics;
use internal::InternalMetrics;
use lazy_static::lazy_static;
use prometheus_client::registry::Registry;

lazy_static! {
    pub static ref INTERNAL_METRICS: InternalMetrics = InternalMetrics::default();
    pub static ref API_METRICS: APIMetrics = APIMetrics::default();
}

/// Register the metrics with the provided registry.
pub fn register_metrics(metrics_registry: &mut Registry) {
    INTERNAL_METRICS.register_metrics(metrics_registry);
    API_METRICS.register_metrics(metrics_registry);
}
