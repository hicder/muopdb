use std::time::Instant;

use anyhow::{Context, Result};
use log::{LevelFilter, info};
use proto::muopdb::SearchRequest;
use proto::muopdb::index_server_client::IndexServerClient;
use serde_json::{json, json_internal};

#[tokio::main]
async fn main() -> Result<()> {
    // Configure logging
    env_logger::Builder::new()
        .filter_level(LevelFilter::Info)
        .format_timestamp_millis()
        .init();

    let addr = String::from("http://127.0.0.1:9002");
    // Create gRPC client
    let mut client = IndexServerClient::connect(addr)
        .await
        .context("Failed to connect to IndexServer")?;

    info!("=========== Starting search ===========");

    // Example query vector (replace with actual input)
    let query = "personal career development";
    // Make an HTTP request to ollama to get the embedding. The body should look like this:
    // {"model": "llama3.2", "prompt": "personal career development"}

    let request_body = json!({
       "model": "nomic-embed-text",
       "prompt": query,
    });

    let http_client = reqwest::Client::new();
    let response = http_client
        .post("http://localhost:11434/api/embeddings")
        .json(&request_body)
        .send()
        .await?;

    let response_body = response.text().await.expect("Failed to read response body");
    let response_map: serde_json::Value = serde_json::from_str(&response_body).unwrap();
    let query_vector_value = response_map["embedding"].as_array().unwrap();
    let query_vector: Vec<f32> = query_vector_value.iter().map(|x| x.as_f64().unwrap()).map(|x| x as f32).collect();

    // Create search request
    let request = tonic::Request::new(SearchRequest {
        index_name: "test-collection-1".to_string(),
        vector: query_vector,
        top_k: 5,
        ef_construction: 100,
        record_metrics: false,
    });

    let start = Instant::now();
    let response = client.search(request).await?;
    let duration = start.elapsed();

    let search_response = response.into_inner();
    info!("Search completed in {:?}", duration);
    info!("Search results:");
    for (id, score) in search_response
        .ids
        .iter()
        .zip(search_response.scores.iter())
    {
        info!("ID: {}, Score: {}", id, score);
    }

    // Read /mnt/muopdb/raw/1m_sentences.txt and print the rows with associated id
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    let file = File::open("/mnt/muopdb/raw/1m_sentences.txt").unwrap();
    let reader = BufReader::new(file);

    let mut id: u64 = 1;
    for line in reader.lines() {
        let line = line.unwrap();
        if search_response.ids.contains(&id) {
            println!("RESULT: {}", line);
        }
        id += 1;
    }

    Ok(())
}
