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
use prometheus_client::encoding::text::encode;
use prometheus_client::metrics::counter::Counter;
use prometheus_client::registry::Registry;
use tokio::net::TcpListener;
use tokio::pin;
use tokio::signal::unix::{signal, SignalKind};

pub struct MetricsServer {
    registry: Registry,
    request_counter: Counter<u64>,
}

impl MetricsServer {
    /// Create a new instance of the metrics server, with a request counter metric.
    pub fn new() -> Self {
        let request_counter: Counter<u64> = Default::default();

        let mut registry = <Registry>::with_prefix("index_server_metrics");

        registry.register(
            "requests",
            "How many requests to the metrics endpoint have been made",
            request_counter.clone(),
        );
        Self {
            registry,
            request_counter,
        }
    }

    /// Start a HTTP server to report metrics.
    pub async fn serve(self, addr: SocketAddr) -> io::Result<()> {
        let registry = Arc::new(self.registry);
        let request_counter = Arc::new(self.request_counter);

        let tcp_listener = TcpListener::bind(addr).await?;
        let server = http1::Builder::new();
        while let Ok((stream, _)) = tcp_listener.accept().await {
            let mut shutdown_stream = signal(SignalKind::terminate())?;
            let io = TokioIo::new(stream);

            let server_clone = server.clone();
            let registry_clone = registry.clone();
            let request_counter_clone = request_counter.clone();
            tokio::task::spawn(async move {
                let conn = server_clone.serve_connection(
                    io,
                    service_fn(make_handler(registry_clone, request_counter_clone)),
                );
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
    registry: Arc<Registry>,
    request_counter: Arc<Counter<u64>>,
) -> impl Fn(Request<Incoming>) -> BoxFuture<'static, io::Result<Response<BoxBody>>> {
    // This closure accepts a request and responds with the OpenMetrics encoding of our metrics.
    move |req: Request<Incoming>| {
        let reg = registry.clone();
        let counter = request_counter.clone();

        Box::pin(async move {
            if req.method() != hyper::Method::GET || req.uri().path() != "/metrics" {
                return Ok(Response::builder()
                    .status(404)
                    .body(full(Bytes::from("Not Found")))
                    .unwrap());
            }

            // Increment the metric counter
            counter.inc();

            let mut buf = String::new();
            encode(&mut buf, &reg.clone())
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
                .map(|_| {
                    let body = full(Bytes::from(buf));
                    Response::builder()
                        .header(
                            hyper::header::CONTENT_TYPE,
                            "application/openmetrics-text; version=1.0.0; charset=utf-8",
                        )
                        .body(body)
                        .unwrap()
                })
        })
    }
}

/// helper function to build a full boxed body
fn full(body: Bytes) -> BoxBody {
    Full::new(body).map_err(|never| match never {}).boxed()
}
