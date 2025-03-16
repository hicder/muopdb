use rdkafka::admin::{AdminClient, AdminOptions, NewTopic, TopicReplication};
use rdkafka::client::DefaultClientContext;
use rdkafka::config::ClientConfig;
use rdkafka::error::KafkaResult;
use rdkafka::util::Timeout;
use std::time::Duration;

pub struct Admin {
    client: AdminClient<DefaultClientContext>,
}

impl Admin {
    pub fn new(brokers: &str) -> Self {
        let client: AdminClient<DefaultClientContext> = ClientConfig::new()
            .set("bootstrap.servers", brokers)
            .create()
            .expect("Admin client creation error");

        Admin { client }
    }

    pub async fn topic_exists(&self, topic: &str) -> KafkaResult<bool> {
        let metadata = self
            .client
            .inner()
            .fetch_metadata(Some(topic), Timeout::After(Duration::from_secs(5)))?;
        Ok(metadata.topics().iter().any(|t| t.name() == topic))
    }

    pub async fn create_topic(&self, topic: &str) -> KafkaResult<()> {
        let new_topic =
            NewTopic::new(topic, 1, TopicReplication::Fixed(3)).set("retention.ms", "-1"); // infinite retention
        let res = self
            .client
            .create_topics(
                &[new_topic],
                &AdminOptions::new()
                    .operation_timeout(Some(Timeout::After(Duration::from_secs(10)))),
            )
            .await?;

        for result in res {
            match result {
                Ok(_) => println!("Topic {} created successfully", topic),
                Err((err, _)) => eprintln!("Failed to create topic {}: {:?}", topic, err),
            }
        }
        Ok(())
    }
}


