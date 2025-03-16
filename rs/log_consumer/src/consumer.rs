use anyhow::Result;
use rdkafka::config::ClientConfig;
use rdkafka::consumer::{BaseConsumer, Consumer};
use rdkafka::error::KafkaError;
use rdkafka::{Message, Offset};
use rmp_serde::decode;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogMessage<T> {
    pub payload: T,
    pub topic: String,
}

pub struct LogConsumer {
    inner: BaseConsumer,
}

impl LogConsumer {
    pub fn new(broker: &str, topic: Option<&str>, group_id: &str) -> Result<Self> {
        let consumer: BaseConsumer = ClientConfig::new()
            .set("bootstrap.servers", broker)
            .set("group.id", group_id)
            // start from the beginning of the topic if no offset is stored
            .set("auto.offset.reset", "earliest")
            // auto commit offsets
            .set("enable.auto.commit", "true")
            .create()?;

        if let Some(topic) = topic {
            consumer.subscribe(&[topic])?;
        }

        Ok(LogConsumer { inner: consumer })
    }

    pub async fn consume_logs(&self) -> Result<usize> {
        // get the stream
        let mut processed_ops = 0;

        while let Some(Ok(result)) = self.inner.poll(Duration::from_secs(1)) {
            processed_ops += 1;
            match decode::from_slice::<LogMessage<Vec<u8>>>(result.payload().unwrap()) {
                Ok(wal_entry_message) => {
                    let decoded: Value = decode::from_slice(&wal_entry_message.payload).unwrap();
                    let message = format!(
                        "WAL offset: {}, partition: {}, payload: {:?}",
                        result.offset(),
                        result.partition(),
                        decoded
                    );
                    println!("{}", message);
                }
                Err(e) => {
                    println!("Failed to decode message: {}", e)
                }
            }
        }
        Ok(processed_ops)
    }

    pub async fn subscribe_to_topics(&self, topics: &[&str]) -> Result<(), KafkaError> {
        self.inner.subscribe(topics)
    }
}


