use std::time::Duration;

use anyhow::Result;
use log::{error, info};
use rdkafka::config::ClientConfig;
use rdkafka::consumer::{BaseConsumer, Consumer};
use rdkafka::{Message, Offset, TopicPartitionList};
use rmp_serde::decode;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogMessage<T> {
    pub payload: T,
    pub topic: String,
}

pub struct LogConsumer {
    inner: BaseConsumer,
}

impl LogConsumer {
    pub fn new(broker: &str) -> Result<Self> {
        let consumer: BaseConsumer = ClientConfig::new()
            .set("bootstrap.servers", broker)
            .set("group.id", "log_consumer")
            // start from the beginning of the topic if no offset is stored
            .set("auto.offset.reset", "earliest")
            // manually commit offsets
            .set("enable.auto.commit", "false")
            .create()?;

        Ok(LogConsumer { inner: consumer })
    }

    pub async fn consume_logs(&self) -> Result<usize> {
        let mut processed_ops = 0;

        while let Some(Ok(result)) = self.inner.poll(Duration::from_millis(500)) {
            processed_ops += 1;
            match decode::from_slice::<LogMessage<Vec<u8>>>(result.payload().unwrap()) {
                Ok(wal_entry_message) => {
                    let decoded: Value = decode::from_slice(&wal_entry_message.payload)?;
                    let message = format!(
                        "WAL offset: {}, partition: {}, payload: {:?}",
                        result.offset(),
                        result.partition(),
                        decoded
                    );
                    info!("{}", message);
                }
                Err(e) => {
                    error!("Failed to decode message: {}", e)
                }
            }
        }

        Ok(processed_ops)
    }

    pub async fn subscribe_to_topic(&self, topic: &str, offset: Option<i64>) -> Result<()> {
        let metadata = self
            .inner
            .fetch_metadata(Some(topic), Duration::from_secs(5))?;

        if metadata.topics().is_empty() {
            return Err(anyhow::anyhow!("Topic {} does not exist", topic));
        }

        let offset = offset.map(Offset::Offset).unwrap_or(Offset::Beginning);

        let topic_metadata = metadata
            .topics()
            .iter()
            .find(|t| t.name() == topic)
            .ok_or_else(|| anyhow::anyhow!("Topic {} metadata not found", topic))?;

        let mut tpl = TopicPartitionList::new();
        match topic_metadata.partitions().get(0) {
            Some(partition) => {
                tpl.add_partition(topic_metadata.name(), partition.id())
                    .set_offset(offset)?;
                self.inner.assign(&tpl)?;
                Ok(())
            }
            None => Err(anyhow::anyhow!("No partitions found for topic {}", topic)),
        }
    }
}

#[cfg(test)]
mod tests {
    use tokio::runtime::Runtime;
    use tokio::time::sleep;

    use super::*;
    use crate::admin::Admin;
    use crate::producer::{LogMessage, LogProducer};

    #[test]
    fn test_log_consumer_creation() {
        // Test that we can create a LogConsumer
        let consumer = LogConsumer::new("localhost:19092,localhost:29092,localhost:39092");
        assert!(consumer.is_ok());
    }

    #[test]
    fn test_consume_logs() {
        let topic = "test_log_consumer";
        let rt = Runtime::new().unwrap();

        // Create producer to send test messages
        let producer = LogProducer::new("localhost:19092,localhost:29092,localhost:39092", topic)
            .expect("Failed to create producer");

        // Create a test message
        let test_data = serde_json::Value::from(serde_json::Map::from_iter([
            ("id".to_string(), serde_json::Value::from(1)),
            ("name".to_string(), serde_json::Value::from("test")),
        ]));

        let log_message = LogMessage {
            payload: rmp_serde::to_vec(&test_data).unwrap(),
            topic: topic.to_string(),
        };

        // Send the test message and consume it
        rt.block_on(async {
            let expected_messages = 5;
            let admin = Admin::new("localhost:19092,localhost:29092,localhost:39092")
                .expect("Failed to create admin");

            if !admin.topic_exists(topic).await.unwrap_or(false) {
                admin
                    .create_topic(topic)
                    .await
                    .expect("Failed to create topic");
                // Give Kafka some time to create the topic
                sleep(Duration::from_secs(1)).await;
            }

            for _ in 0..expected_messages {
                producer
                    .send_logs(&log_message)
                    .await
                    .expect("Failed to send logs");
            }

            sleep(Duration::from_secs(2)).await;

            let consumer = LogConsumer::new("localhost:19092,localhost:29092,localhost:39092")
                .expect("Failed to create consumer");

            consumer
                .subscribe_to_topic(topic, None)
                .await
                .expect("Failed to subscribe to topic");

            let start_time = std::time::Instant::now();
            let mut processed = 0;
            // consume logs for 10 seconds
            while start_time.elapsed() < Duration::from_secs(10) {
                processed += consumer
                    .consume_logs()
                    .await
                    .expect("Failed to consume logs");
                sleep(Duration::from_millis(100)).await;
            }

            assert!(
                processed >= expected_messages,
                "Expected to process at least {} messages",
                expected_messages
            );
        });
    }
}
