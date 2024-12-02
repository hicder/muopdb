use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::quantization::Quantizer;

pub struct NoQuantizer {
    dimension: usize,
}

impl NoQuantizer {
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

impl Quantizer for NoQuantizer {
    fn quantize(&self, _value: &[f32]) -> Vec<u8> {
        // Throw an error if called
        todo!()
    }

    fn quantized_dimension(&self) -> usize {
        self.dimension
    }

    fn original_vector(&self, _quantized_vector: &[u8]) -> Vec<f32> {
        // Throw an error if called
        todo!()
    }

    fn distance(
        &self,
        _query: &[u8],
        _point: &[u8],
        _implem: utils::distance::l2::L2DistanceCalculatorImpl,
    ) -> f32 {
        // Throw an error if called
        todo!()
    }

    fn read(dir: String) -> Result<Self>
    where
        Self: Sized,
    {
        let reader = NoQuantizerReader::new(dir);
        reader.read()
    }
}

#[derive(Serialize, Deserialize)]
pub struct NoQuantizerConfig {
    dimension: usize,
}

pub struct NoQuantizerReader {
    base_directory: String,
}

impl NoQuantizerReader {
    pub fn new(base_directory: String) -> Self {
        Self { base_directory }
    }

    pub fn read(&self) -> Result<NoQuantizer> {
        // Deserialieze the config
        let config = serde_yaml::from_str::<NoQuantizerConfig>(&std::fs::read_to_string(
            &format!("{}/no_op_quantizer_config.yaml", self.base_directory),
        )?)?;
        Ok(NoQuantizer::new(config.dimension))
    }
}

// Writer

pub struct NoQuantizerWriter {
    base_directory: String,
}

impl NoQuantizerWriter {
    pub fn new(base_directory: String) -> Self {
        Self { base_directory }
    }

    pub fn write(&self, quantizer: &NoQuantizer) -> Result<()> {
        let config = NoQuantizerConfig {
            dimension: quantizer.dimension,
        };
        std::fs::write(
            &format!("{}/no_op_quantizer_config.yaml", self.base_directory),
            serde_yaml::to_string(&config)?,
        )?;
        Ok(())
    }
}
