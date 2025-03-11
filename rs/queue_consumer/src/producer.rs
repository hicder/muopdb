use rdkafka::config::ClientConfig;
use rdkafka::producer::{FutureProducer, FutureRecord};
use rdkafka::util::Timeout;
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueMessage<T> {
    pub payload: T,
    pub topic: String,
}
pub struct QueueProducer {
    producer: FutureProducer,
    topic: String,
}

impl QueueProducer {
    pub fn new(brokers: &str, topic: &str) -> Self {
        let producer: FutureProducer = ClientConfig::new()
            .set("bootstrap.servers", brokers)
            .create()
            .expect("Producer creation failed");

        QueueProducer {
            producer,
            topic: topic.to_string(),
        }
    }

    pub async fn send_message(&self, message: &QueueMessage<String>) {
        let payload = serde_json::to_string(&message)
            .expect("Failed to serialize message");

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