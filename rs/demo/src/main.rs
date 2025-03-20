use std::time::Instant;

use anyhow::{Context, Result};
use hdf5::File;
use log::{info, LevelFilter};
use log_consumer::admin::Admin;
use log_consumer::producer::{LogMessage, LogProducer};
use ndarray::s;
use proto::muopdb::index_server_client::IndexServerClient;
use proto::muopdb::{FlushRequest, Id, InsertPackedRequest};
use rmp_serde::encode;

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

    let brokers = "localhost:19092,localhost:29092,localhost:39092";
    let topic = "wal_topic-test-collection-1";
    let producer = LogProducer::new(&brokers, &topic)?;
    let admin = Admin::new(&brokers)?;

    if !admin.topic_exists(&topic).await? {
        admin.create_topic(&topic).await?;
    }

    info!("=========== Inserting documents ===========");

    // Read embeddings from HDF5 file
    let file = File::open("/mnt/muopdb/raw/1m_embeddings.hdf5")?;
    let dataset = file.dataset("embeddings")?;
    let embeddings = dataset.read_2d::<f32>()?;

    // Insert embeddings in batches into MuopDB
    let batch_size = 100_000;
    let total_embeddings = embeddings.nrows();
    let mut start_idx = 0;

    let mut start = Instant::now();
    while start_idx < total_embeddings {
        let end_idx = (start_idx + batch_size).min(total_embeddings);
        let batch = &embeddings.slice(s![start_idx..end_idx, ..]);

        let mut vectors = Vec::with_capacity(batch.len() * 768);
        for row in batch.rows() {
            vectors.extend(row.iter().map(|&v| v as f32));
        }

        // Generate IDs
        let ids: Vec<u128> = (start_idx + 1..=end_idx).map(|i| i as u128).collect();
        let id_buffer = utils::mem::transmute_slice_to_u8(&ids);
        let vector_buffer = utils::mem::transmute_slice_to_u8(&vectors);

        // Create and send insert request
        let request = tonic::Request::new(InsertPackedRequest {
            collection_name: "test-collection-1".to_string(),
            doc_ids: id_buffer.to_vec(),
            vectors: vector_buffer.to_vec(),
            user_ids: vec![Id {
                low_id: 0,
                high_id: 0,
            }],
        });

        let msgpack_payload =
            encode::to_vec(&request.get_ref().clone()).expect("Failed to serialize request");

        let msg = LogMessage {
            payload: msgpack_payload,
            topic: topic.to_string(),
        };

        producer.send_logs(&msg).await?;

        client.insert_packed(request).await?;
        start_idx = end_idx;
    }

    let mut duration = start.elapsed();
    info!("Inserted all documents in {:?}", duration);

    // Done inserting, now start indexing.
    info!("Start indexing documents...");
    start = Instant::now();
    let request = tonic::Request::new(FlushRequest {
        collection_name: "test-collection-1".to_string(),
    });
    client.flush(request).await?;
    duration = start.elapsed();
    info!("Indexing documents completed in {:?}", duration);

    Ok(())
}
