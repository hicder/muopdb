use opentelemetry::global;
use opentelemetry::propagation::Injector;
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::trace::{Sampler, TracerProvider};
use opentelemetry_sdk::Resource;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

pub struct TracingConfig {
    pub service_name: String,
    pub otlp_endpoint: String,
    pub sampling_rate: f64,
    pub enabled: bool,
}

pub fn init_tracing(config: &TracingConfig) -> anyhow::Result<()> {
    if !config.enabled {
        tracing_subscriber::fmt()
            .with_env_filter(EnvFilter::from_default_env())
            .init();
        return Ok(());
    }

    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_tonic()
        .with_endpoint(&config.otlp_endpoint)
        .build()?;

    let provider = TracerProvider::builder()
        .with_batch_exporter(exporter, opentelemetry_sdk::runtime::Tokio)
        .with_sampler(Sampler::TraceIdRatioBased(config.sampling_rate))
        .with_resource(Resource::new(vec![opentelemetry::KeyValue::new(
            "service.name",
            config.service_name.clone(),
        )]))
        .build();

    global::set_tracer_provider(provider.clone());
    let tracer = provider.tracer("muopdb");

    let telemetry = tracing_opentelemetry::layer().with_tracer(tracer);
    let env_filter = EnvFilter::from_default_env();
    let fmt_layer = tracing_subscriber::fmt::layer();

    tracing_subscriber::registry()
        .with(env_filter)
        .with(telemetry)
        .with(fmt_layer)
        .init();

    Ok(())
}

pub fn shutdown_tracing() {
    global::shutdown_tracer_provider();
}

pub struct MetadataInjector<'a>(pub &'a mut tonic::metadata::MetadataMap);

impl<'a> Injector for MetadataInjector<'a> {
    fn set(&mut self, key: &str, value: String) {
        if let Ok(key) = tonic::metadata::MetadataKey::from_bytes(key.as_bytes()) {
            if let Ok(value) = tonic::metadata::MetadataValue::try_from(&value) {
                self.0.insert(key, value);
            }
        }
    }
}

pub struct MetadataExtractor<'a>(pub &'a tonic::metadata::MetadataMap);

impl<'a> opentelemetry::propagation::Extractor for MetadataExtractor<'a> {
    fn get(&self, key: &str) -> Option<&str> {
        self.0.get(key).and_then(|v| v.to_str().ok())
    }

    fn keys(&self) -> Vec<&str> {
        self.0
            .keys()
            .map(|k| match k {
                tonic::metadata::KeyRef::Ascii(k) => k.as_str(),
                tonic::metadata::KeyRef::Binary(k) => k.as_str(),
            })
            .collect()
    }
}
