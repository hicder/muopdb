use config::collection::CollectionConfig;
use config::enums::QuantizerType;
use index::collection::reader::CollectionReader;
use index::collection::BoxedCollection;
use quantization::noq::NoQuantizer;
use quantization::pq::ProductQuantizer;
use utils::distance::l2::L2DistanceCalculator;
use utils::file_io::env::EnvConfig;

pub struct CollectionProvider {
    data_directory: String,
    env_config: Option<EnvConfig>,
}

impl CollectionProvider {
    pub fn new(data_directory: String, env_config: Option<EnvConfig>) -> Self {
        Self {
            data_directory,
            env_config,
        }
    }

    pub async fn read_collection(&self, name: &str) -> Option<BoxedCollection> {
        let collection_path = format!("{}/{}", self.data_directory, name);
        let reader = CollectionReader::new(
            name.to_string(),
            collection_path.clone(),
            self.env_config.clone(),
        );

        // Read the collection config
        let config_path = format!("{}/collection_config.json", collection_path);
        let config = std::fs::read_to_string(config_path).unwrap();
        let collection_config: CollectionConfig = serde_json::from_str(&config).unwrap();

        match collection_config.quantization_type {
            QuantizerType::NoQuantizer => {
                match reader.read::<NoQuantizer<L2DistanceCalculator>>().await {
                    Ok(collection) => Some(BoxedCollection::CollectionNoQuantizationL2(collection)),
                    Err(e) => {
                        println!("Failed to read collection: {}", e);
                        None
                    }
                }
            }
            QuantizerType::ProductQuantizer => {
                match reader
                    .read::<ProductQuantizer<L2DistanceCalculator>>()
                    .await
                {
                    Ok(collection) => {
                        Some(BoxedCollection::CollectionProductQuantization(collection))
                    }
                    Err(e) => {
                        println!("Failed to read collection: {}", e);
                        None
                    }
                }
            }
        }
    }

    pub fn data_directory(&self) -> &str {
        &self.data_directory
    }
}
