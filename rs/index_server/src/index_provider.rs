use index::hnsw::reader::HnswReader;
use index::index::BoxedSearchable;
use index::ivf::reader::IvfReader;
use index::spann::reader::SpannReader;
use index_writer::config::{BaseConfig, DistanceType, QuantizerConfig, QuantizerType};
use quantization::noq::noq::NoQuantizer;
use quantization::pq::pq::ProductQuantizer;
use utils::distance::dot_product::DotProductDistanceCalculator;
use utils::distance::l2::L2DistanceCalculator;

pub struct IndexProvider {
    data_directory: String,
}

impl IndexProvider {
    pub fn new(data_directory: String) -> Self {
        Self { data_directory }
    }

    pub fn read_index(&self, name: &str) -> Option<BoxedSearchable> {
        let index_path = format!("{}/{}", self.data_directory, name);
        let base_config_path = format!("{}/base_config.yaml", index_path);
        let quantizer_config_path = format!("{}/quantizer_config.yaml", index_path);

        let base_config: BaseConfig =
            serde_yaml::from_reader(std::fs::File::open(base_config_path).unwrap()).unwrap();

        let quantizer_config: QuantizerConfig =
            serde_yaml::from_reader(std::fs::File::open(quantizer_config_path).unwrap()).unwrap();

        match base_config.index_type {
            index_writer::config::IndexType::Hnsw => {
                let reader = HnswReader::new(index_path);
                match quantizer_config.quantizer_type {
                    QuantizerType::ProductQuantizer => match reader.read::<ProductQuantizer>() {
                        Ok(index) => Some(Box::new(index)),
                        Err(e) => {
                            println!("Failed to read ProductQuantizer index: {}", e);
                            None
                        }
                    },
                    QuantizerType::NoQuantizer => match reader.read::<NoQuantizer>() {
                        Ok(index) => Some(Box::new(index)),
                        Err(e) => {
                            println!("Failed to read NoQuantizer index: {}", e);
                            None
                        }
                    },
                }
            }
            index_writer::config::IndexType::Ivf => {
                let reader = IvfReader::new(index_path);
                match quantizer_config.quantizer_type {
                    QuantizerType::ProductQuantizer => match base_config.index_distance_type {
                        DistanceType::DotProduct => match reader.read::<ProductQuantizer, DotProductDistanceCalculator>() {
                            Ok(index) => Some(Box::new(index)),
                            Err(e) => {
                                println!("Failed to read ProductQuantizer index: {}", e);
                                None
                            }
                        },
                        DistanceType::L2 => match reader.read::<ProductQuantizer, L2DistanceCalculator>() {
                            Ok(index) => Some(Box::new(index)),
                            Err(e) => {
                                println!("Failed to read ProductQuantizer index: {}", e);
                                None
                            }
                        },           
                    },
                    QuantizerType::NoQuantizer => match base_config.index_distance_type {
                        DistanceType::DotProduct => match reader.read::<NoQuantizer, DotProductDistanceCalculator>() {
                            Ok(index) => Some(Box::new(index)),
                            Err(e) => {
                                println!("Failed to read ProductQuantizer index: {}", e);
                                None
                            }
                        },
                        DistanceType::L2 => match reader.read::<NoQuantizer, L2DistanceCalculator>() {
                            Ok(index) => Some(Box::new(index)),
                            Err(e) => {
                                println!("Failed to read ProductQuantizer index: {}", e);
                                None
                            }
                        },
                    }

                }
            }
            index_writer::config::IndexType::Spann => {
                let reader = SpannReader::new(index_path);
                match reader.read() {
                    Ok(index) => Some(Box::new(index)),
                    Err(e) => {
                        println!("Failed to read index: {}", e);
                        None
                    }
                }
            }
        }
    }
}
