use std::time::Duration;

use anyhow::Result;
use rdkafka::admin::{AdminClient, AdminOptions, NewTopic, TopicReplication};
use rdkafka::client::DefaultClientContext;
use rdkafka::config::ClientConfig;
use rdkafka::util::Timeout;

pub struct Admin {
    client: AdminClient<DefaultClientContext>,
}

impl Admin {
    pub fn new(brokers: &str) -> Result<Self> {
        let client: AdminClient<DefaultClientContext> = ClientConfig::new()
            .set("bootstrap.servers", brokers)
            .create()?;

        Ok(Admin { client })
    }

    pub async fn topic_exists(&self, topic: &str) -> Result<bool> {
        let metadata = self
            .client
            .inner()
            .fetch_metadata(Some(topic), Timeout::After(Duration::from_secs(5)))
            .map_err(|e| anyhow::anyhow!("Failed to fetch metadata: {:?}", e))?;
        Ok(metadata.topics().iter().any(|t| t.name() == topic))
    }

    pub async fn create_topic(&self, topic: &str) -> Result<()> {
        let new_topic =
            NewTopic::new(topic, 1, TopicReplication::Fixed(3)).set("retention.ms", "-1"); // infinite retention
        let res = self
            .client
            .create_topics(
                &[new_topic],
                &AdminOptions::new()
                    .operation_timeout(Some(Timeout::After(Duration::from_secs(10)))),
            )
            .await
            .map_err(|e| anyhow::anyhow!("Failed to create topic: {:?}", e))?;

        for result in res {
            match result {
                Ok(_) => println!("Topic {} created successfully", topic),
                Err((err, _)) => eprintln!("Failed to create topic {}: {:?}", topic, err),
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use tokio::runtime::Runtime;

    use super::*;

    #[test]
    fn test_admin_creation() {
        // Test that we can create an Admin client
        let admin = Admin::new("localhost:19092,localhost:29092,localhost:39092");
        assert!(admin.is_ok());
    }

    #[test]
    fn test_create_topic() {
        let rt = Runtime::new().unwrap();

        rt.block_on(async {
            let admin = Admin::new("localhost:19092,localhost:29092,localhost:39092")
                .expect("Failed to create admin client");

            let random_topic = format!(
                "test_admin_create_topic_{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs()
            );

            // Create the topic
            let result = admin.create_topic(&random_topic).await;
            assert!(result.is_ok(), "Topic creation failed");

            // Verify the topic exists
            let exists = admin
                .topic_exists(&random_topic)
                .await
                .expect("Failed to check topic");
            assert!(exists, "Topic should exist after creation");
        });
    }
}
