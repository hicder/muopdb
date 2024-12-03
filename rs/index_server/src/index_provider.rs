use index::hnsw::reader::HnswReader;
use index::index::BoxedIndex;
use quantization::pq::ProductQuantizer;

pub struct IndexProvider {
    data_directory: String,
}

impl IndexProvider {
    pub fn new(data_directory: String) -> Self {
        Self { data_directory }
    }

    pub fn read_index(&self, name: &str) -> Option<BoxedIndex> {
        let index_path = format!("{}/{}", self.data_directory, name);
        let reader = HnswReader::new(index_path);
        match reader.read::<ProductQuantizer>() {
            Ok(index) => Some(Box::new(index)),
            Err(e) => {
                println!("Failed to read index: {}", e);
                None
            }
        }
    }
}
