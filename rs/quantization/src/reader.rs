use std::path::Path;

use anyhow::Result;

use crate::noq::noq::{NoQuantizer, NoQuantizerReader};
use crate::pq::pq::{ProductQuantizer, ProductQuantizerReader};

pub struct QuantizationReader {
    base_directory: String,
}

pub enum QuantizationType {
    ProductQuantizer,
    NoQuantization,
}

impl QuantizationReader {
    pub fn new(base_directory: String) -> Self {
        Self { base_directory }
    }

    pub fn get_quantization_type(&self) -> QuantizationType {
        // If exists the file, then it is a product quantizer
        if Path::new(&self.base_directory)
            .join("product_quantizer_config.yaml")
            .exists()
        {
            return QuantizationType::ProductQuantizer;
        }
        QuantizationType::NoQuantization
    }

    pub fn read_product_quantizer(&self) -> Result<ProductQuantizer> {
        let reader = ProductQuantizerReader::new(self.base_directory.clone());
        reader.read()
    }

    pub fn read_no_quantization(&self) -> Result<NoQuantizer> {
        let reader = NoQuantizerReader::new(self.base_directory.clone());
        reader.read()
    }
}
