use index::hnsw::reader::HnswReader;
use index::index::BoxedIndex;
use index::ivf::reader::IvfReader;
use index::spann::reader::SpannReader;
use index_writer::config::BaseConfig;
use quantization::noq::noq::NoQuantizer;
use quantization::pq::pq::ProductQuantizer;

pub struct IndexProvider {
    data_directory: String,
}

impl IndexProvider {
    pub fn new(data_directory: String) -> Self {
        Self { data_directory }
    }

    pub fn read_index(&self, name: &str) -> Option<BoxedIndex> {
        let index_path = format!("{}/{}", self.data_directory, name);
        let index_writer_config_path = format!("{}/base_config.yaml", index_path);

        let base_config: BaseConfig =
            serde_yaml::from_reader(std::fs::File::open(index_writer_config_path).unwrap())
                .unwrap();

        match base_config.index_type {
            index_writer::config::IndexType::Hnsw => {
                let reader = HnswReader::new(index_path);
                match reader.read::<ProductQuantizer>() {
                    Ok(index) => Some(Box::new(index)),
                    Err(e) => {
                        println!("Failed to read index: {}", e);
                        None
                    }
                }
            }
            index_writer::config::IndexType::Ivf => {
                let reader = IvfReader::new(index_path);
                match reader.read::<NoQuantizer>() {
                    Ok(index) => Some(Box::new(index)),
                    Err(e) => {
                        println!("Failed to read index: {}", e);
                        None
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
