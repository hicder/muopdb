use lazy_static::lazy_static;
use prometheus_client::registry::Registry;

mod internal;
use internal::InternalMetrics;

lazy_static! {
    pub static ref INTERNAL_METRICS: InternalMetrics = InternalMetrics::default();
}

/// Register the metrics with the provided registry.
pub fn register_metrics(metrics_registry: &mut Registry) {
    INTERNAL_METRICS.register_metrics(metrics_registry);
}
