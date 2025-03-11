use rdkafka::config::ClientConfig;
use rdkafka::consumer::{Consumer, StreamConsumer};
use rdkafka::Message;
use serde::{Deserialize, Serialize};
use tokio_stream::StreamExt;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueMessage<T> {
    pub payload: T,
    pub topic: String,
}

pub struct QueueConsumer {
    inner: StreamConsumer
}

impl QueueConsumer {
    pub fn new(broker: &str, topic: &str, group_id: &str) -> Self {
        let consumer: StreamConsumer = ClientConfig::new()
            .set("bootstrap.servers", broker)
            .set("group.id", group_id)
            .set("auto.offset.reset", "earliest")
            .create()
            .expect("Consumer creation failed");

        consumer
            .subscribe(&[topic])
            .expect("Subscribing to topic failed");


        QueueConsumer {
            inner: consumer
        }
    }

    pub async fn consume_messages(&self) {
        // get the stream
        let mut stream = self.inner.stream();

        while let Some(result) = stream.next().await {
            match result {
                Ok(message) => {
                    let payload = match message.payload_view::<str>() {
                        Some(Ok(payload)) => {
                            payload
                        }
                        Some(Err(e)) => {
                            eprintln!("Error while deserializing message payload: {:?}", e);
                            continue;
                        }
                        None => {
                            eprintln!("Failed to get message payload");
                            continue;
                        }
                    };

                    match serde_json::from_str::<QueueMessage<String>>(payload) {
                        Ok(wal_entry_message) => {
                            let message =
                                format!("WAL offset: {}, partition: {}, payload: {}",
                                        message.offset(),
                                        message.partition(),
                                        wal_entry_message.payload
                                );
                            println!("{}", message);
                        }
                        Err(e) => {
                            eprintln!("Error while deserializing message payload: {:?}", e);
                            continue;
                        }
                    }
                }

                Err(error) => {
                    eprint!("Panda error: {}", error)
                }
            }
        }
    }
}


