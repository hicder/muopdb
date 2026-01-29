use std::net::SocketAddr;
use std::sync::Arc;
use std::{fs, io};

use futures::future::BoxFuture;
use http_body_util::{combinators, BodyExt, Full};
use hyper::body::{Bytes, Incoming};
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Request, Response};
use hyper_util::rt::TokioIo;
use metrics::{register_metrics, INTERNAL_METRICS};
use pprof::protos::Message;
use prometheus_client::encoding::text::encode;
use prometheus_client::registry::Registry;
use tokio::net::TcpListener;
use tokio::pin;
use tokio::signal::unix::{signal, SignalKind};

pub struct HttpServer {
    metrics_registry: Arc<Registry>,
    profile_output_dir: Option<String>,
}

impl HttpServer {
    pub fn new(profile_output_dir: Option<String>) -> Self {
        let mut metrics_registry: Registry = <Registry>::default();
        register_metrics(&mut metrics_registry);
        Self {
            metrics_registry: Arc::new(metrics_registry),
            profile_output_dir,
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
            let profile_output_dir_clone = self.profile_output_dir.clone();
            tokio::task::spawn(async move {
                let conn = server_clone.serve_connection(
                    io,
                    service_fn(make_handler(
                        metrics_registry_clone,
                        profile_output_dir_clone,
                    )),
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
    metrics_registry: Arc<Registry>,
    profile_output_dir: Option<String>,
) -> impl Fn(Request<Incoming>) -> BoxFuture<'static, io::Result<Response<BoxBody>>> {
    move |req: Request<Incoming>| {
        let reg = metrics_registry.clone();
        let profile_dir = profile_output_dir.clone();

        Box::pin(async move {
            let path = req.uri().path();

            // Handle metrics endpoint
            if path == "/metrics" {
                if req.method() != hyper::Method::GET {
                    return Ok(Response::builder()
                        .status(405)
                        .body(full(Bytes::from("Method Not Allowed")))
                        .expect("Failed to build response"));
                }

                log::info!("Received metrics request");
                INTERNAL_METRICS.prometheus_requests.inc();

                let mut buf = String::new();
                return encode(&mut buf, &reg)
                    .map_err(std::io::Error::other)
                    .map(|_| {
                        let body = full(Bytes::from(buf));
                        Response::builder()
                            .header(
                                hyper::header::CONTENT_TYPE,
                                "application/openmetrics-text; version=1.0.0; charset=utf-8",
                            )
                            .body(body)
                            .expect("Failed to build response")
                    });
            }

            // Handle profile endpoint
            if path == "/debug/pprof/profile" {
                if req.method() != hyper::Method::GET {
                    return Ok(Response::builder()
                        .status(405)
                        .body(full(Bytes::from("Method Not Allowed")))
                        .expect("Failed to build response"));
                }

                return handle_profile_request(&profile_dir, req.uri().query()).await;
            }

            // 404 for all other paths
            Ok(Response::builder()
                .status(404)
                .body(full(Bytes::from("Not Found")))
                .expect("Failed to build response"))
        })
    }
}

/// Helper function to build a full boxed body
fn full(body: Bytes) -> BoxBody {
    Full::new(body).map_err(|never| match never {}).boxed()
}

/// Handle profile generation request
async fn handle_profile_request(
    profile_output_dir: &Option<String>,
    query: Option<&str>,
) -> io::Result<Response<BoxBody>> {
    if profile_output_dir.is_none() {
        let response_json =
            r#"{"error":"Profiling is not enabled. Set --profile-output-dir to enable."}"#;
        return Ok(Response::builder()
            .status(404)
            .header(hyper::header::CONTENT_TYPE, "application/json")
            .body(full(Bytes::from(response_json)))
            .expect("Failed to build response"));
    }

    let profile_dir = profile_output_dir.as_ref().unwrap();

    ensure_directory_exists(profile_dir)?;

    let duration = parse_duration_from_query(query)?;

    log::info!("Starting CPU profiling for {} seconds", duration.as_secs());

    match collect_cpu_profile(profile_dir, duration).await {
        Ok(profile_path) => {
            let response_json = format!(
                r#"{{"profile_path":"{}","duration_seconds":{}}}"#,
                profile_path,
                duration.as_secs()
            );
            Ok(Response::builder()
                .status(200)
                .header(hyper::header::CONTENT_TYPE, "application/json")
                .body(full(Bytes::from(response_json)))
                .expect("Failed to build response"))
        }
        Err(e) => {
            log::error!("Profile generation failed: {}", e);
            let error_msg = e.to_string().replace('"', r#"\""#);
            let response_json = format!(r#"{{"error":"{}"}}"#, error_msg);
            Ok(Response::builder()
                .status(500)
                .header(hyper::header::CONTENT_TYPE, "application/json")
                .body(full(Bytes::from(response_json)))
                .expect("Failed to build response"))
        }
    }
}

/// Ensure the profile output directory exists
fn ensure_directory_exists(path: &str) -> io::Result<()> {
    if !std::path::Path::new(path).exists() {
        fs::create_dir_all(path)?;
        log::info!("Created profile output directory: {}", path);
    }
    Ok(())
}

/// Parse duration from query parameter
fn parse_duration_from_query(query: Option<&str>) -> io::Result<std::time::Duration> {
    let seconds = match query {
        Some(q) => {
            let params: std::collections::HashMap<String, String> =
                url::form_urlencoded::parse(q.as_bytes())
                    .into_owned()
                    .collect();
            params
                .get("seconds")
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(30)
        }
        None => 30,
    };

    if !(1..=300).contains(&seconds) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Duration must be between 1 and 300 seconds",
        ));
    }

    Ok(std::time::Duration::from_secs(seconds))
}

/// Collect CPU profile using pprof
async fn collect_cpu_profile(
    output_dir: &str,
    duration: std::time::Duration,
) -> io::Result<String> {
    let guard = pprof::ProfilerGuardBuilder::default()
        .frequency(100)
        .blocklist(&["libc", "libgcc", "pthread", "vdso"])
        .build()
        .map_err(|e| io::Error::other(format!("Failed to start profiler: {}", e)))?;

    log::info!("Collecting profile for {} seconds...", duration.as_secs());
    tokio::time::sleep(duration).await;

    let report = guard
        .report()
        .build()
        .map_err(|e| io::Error::other(format!("Failed to build profile report: {}", e)))?;

    let profile = report
        .pprof()
        .map_err(|e| io::Error::other(format!("Failed to generate protobuf profile: {}", e)))?;

    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| io::Error::other(format!("Failed to get timestamp: {}", e)))?
        .as_secs();
    let filename = format!("{}/profile-{}.pb", output_dir, timestamp);

    let mut content = Vec::new();
    profile
        .encode(&mut content)
        .map_err(|e| io::Error::other(format!("Failed to encode profile protobuf: {}", e)))?;

    fs::write(&filename, content)?;

    log::info!("Profile written to {}", filename);
    Ok(filename)
}
