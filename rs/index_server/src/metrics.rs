use std::io;
use std::net::SocketAddr;
use std::sync::Arc;

use futures::future::BoxFuture;
use http_body_util::{combinators, BodyExt, Full};
use hyper::body::{Bytes, Incoming};
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Request, Response};
use hyper_util::rt::TokioIo;
use lazy_static::lazy_static;
use prometheus_client::encoding::text::encode;
use prometheus_client::metrics::counter::Counter;
use prometheus_client::registry::Registry;
use tokio::net::TcpListener;
use tokio::pin;
use tokio::signal::unix::{signal, SignalKind};

lazy_static! {
    /// Add your own metrics here. Remember to add them to the `register_metrics` function.
    /// Example: a counter metric for the number of incoming requests to the metrics endpoint.
    pub static ref METRICS_REQUESTS: Counter<u64> = Counter::default();
    pub static ref NUM_COLLECTIONS: Counter<u64> = Counter::default();
}

/// Register the metrics with the provided registry.
fn register_metrics(metrics_registry: &mut Registry) {
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
}

pub struct HttpServer {
    metrics_registry: Arc<Registry>,
}

impl HttpServer {
    pub fn new() -> Self {
        let mut metrics_registry: Registry = <Registry>::default();
        register_metrics(&mut metrics_registry);
        Self {
            metrics_registry: Arc::new(metrics_registry),
        }
    }

    /// Start a HTTP server.
    pub async fn serve(&self, addr: SocketAddr) -> io::Result<()> {
        let tcp_listener = TcpListener::bind(addr).await?;
        let server = http1::Builder::new();
        while let Ok((stream, _)) = tcp_listener.accept().await {
            let mut shutdown_stream = signal(SignalKind::terminate())?;
            let io = TokioIo::new(stream);

            let server_clone = server.clone();
            let metrics_registry_clone = self.metrics_registry.clone();
            tokio::task::spawn(async move {
                let conn = server_clone
                    .serve_connection(io, service_fn(make_handler(metrics_registry_clone)));
                pin!(conn);
                tokio::select! {
                    _ = conn.as_mut() => {}
                    _ = shutdown_stream.recv() => {
                        conn.as_mut().graceful_shutdown();
                    }
                }
            });
        }
        Ok(())
    }
}

/// Boxed HTTP body for responses
type BoxBody = combinators::BoxBody<Bytes, hyper::Error>;

/// This function returns a HTTP handler
fn make_handler(
    metrics_registry: Arc<Registry>,
) -> impl Fn(Request<Incoming>) -> BoxFuture<'static, io::Result<Response<BoxBody>>> {
    // This closure accepts a request and responds with the OpenMetrics encoding of our metrics.
    move |req: Request<Incoming>| {
        let reg = metrics_registry.clone();

        Box::pin(async move {
            if req.method() != hyper::Method::GET || req.uri().path() != "/metrics" {
                return Ok(Response::builder()
                    .status(404)
                    .body(full(Bytes::from("Not Found")))
                    .expect("Failed to build response"));
            }

            // Increment the metric counter
            log::info!("Received metrics request");
            METRICS_REQUESTS.inc();

            let mut buf = String::new();
            encode(&mut buf, &reg)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
                .map(|_| {
                    let body = full(Bytes::from(buf));
                    Response::builder()
                        .header(
                            hyper::header::CONTENT_TYPE,
                            "application/openmetrics-text; version=1.0.0; charset=utf-8",
                        )
                        .body(body)
                        .expect("Failed to build response")
                })
        })
    }
}

/// Helper function to build a full boxed body
fn full(body: Bytes) -> BoxBody {
    Full::new(body).map_err(|never| match never {}).boxed()
}
