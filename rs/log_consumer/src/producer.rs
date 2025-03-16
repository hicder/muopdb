use rdkafka::config::ClientConfig;
use rdkafka::producer::{FutureProducer, FutureRecord};
use rdkafka::util::Timeout;
use rmp_serde::encode;
use serde::{Deserialize, Serialize};
use std::time::Duration;

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
    pub fn new(brokers: &str, topic: &str) -> Self {
        let producer: FutureProducer = ClientConfig::new()
            .set("bootstrap.servers", brokers)
            .create()
            .expect("Producer creation failed");

        LogProducer {
            producer,
            topic: topic.to_string(),
        }
    }

    pub async fn send_logs(&self, message: &LogMessage<Vec<u8>>) {
        let payload = encode::to_vec(&message).expect("Failed to serialize message");

        self.producer
            .send(
                FutureRecord::to(&self.topic)
                    .payload(&payload)
                    .key(&self.topic),
                Timeout::After(Duration::from_secs(0)),
            )
            .await
            .unwrap();
    }
}

