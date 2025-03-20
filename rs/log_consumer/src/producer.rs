use std::time::Duration;

use anyhow::Result;
use rdkafka::config::ClientConfig;
use rdkafka::producer::{FutureProducer, FutureRecord};
use rdkafka::util::Timeout;
use rmp_serde::encode;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogMessage<T> {
    pub payload: T,
    pub topic: String,
}
pub struct LogProducer {
    producer: FutureProducer,
    topic: String,
}

impl LogProducer {
    pub fn new(brokers: &str, topic: &str) -> Result<Self> {
        let producer: FutureProducer = ClientConfig::new()
            .set("bootstrap.servers", brokers)
            .create()?;

        Ok(LogProducer {
            producer,
            topic: topic.to_string(),
        })
    }

    pub async fn send_logs(&self, message: &LogMessage<Vec<u8>>) -> Result<()> {
        let payload = encode::to_vec(&message).expect("Failed to serialize message");

        self.producer
            .send(
                FutureRecord::to(&self.topic)
                    .payload(&payload)
                    .key(&self.topic),
                Timeout::After(Duration::from_secs(0)),
            )
            .await
            .map_err(|e| anyhow::anyhow!("Failed to send message: {:?}", e))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use serde_json::Value;
    use tokio::runtime::Runtime;

    use super::*;

    #[test]
    fn test_log_producer_creation() {
        // Test that we can create a LogProducer
        let producer = LogProducer::new(
            "localhost:19092,localhost:29092,localhost:39092",
            "test_topic",
        );
        assert!(producer.is_ok());
    }

    #[test]
    fn test_send_logs() {
        let rt = Runtime::new().unwrap();

        rt.block_on(async {
            // Create a producer
            let producer = LogProducer::new(
                "localhost:19092,localhost:29092,localhost:39092",
                "test_topic",
            )
            .expect("Failed to create producer");

            // Create a test message
            let test_data = Value::from(serde_json::Map::from_iter([
                ("id".to_string(), Value::from(1)),
                ("name".to_string(), Value::from("test")),
            ]));

            let serialized = rmp_serde::to_vec(&test_data).expect("Failed to serialize test data");

            let message = LogMessage {
                payload: serialized,
                topic: "test_topic".to_string(),
            };

            // Send the test message
            let result = producer.send_logs(&message).await;
            assert!(result.is_ok());
        });
    }
}
