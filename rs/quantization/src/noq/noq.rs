use std::marker::PhantomData;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use utils::distance::l2::L2DistanceCalculator;
use utils::DistanceCalculator;

use crate::quantization::{Quantizer, WritableQuantizer};

#[derive(Debug, Clone, Copy)]
pub struct NoQuantizer<D: DistanceCalculator> {
    dimension: usize,

    _marker: PhantomData<D>,
}

impl<D: DistanceCalculator> NoQuantizer<D> {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            _marker: PhantomData,
        }
    }
}

impl<D: DistanceCalculator> Quantizer for NoQuantizer<D> {
    type QuantizedT = f32;

    fn quantize(&self, value: &[f32]) -> Vec<f32> {
        value.to_vec()
    }

    fn quantized_dimension(&self) -> usize {
        self.dimension
    }

    fn original_vector(&self, quantized_vector: &[f32]) -> Vec<f32> {
        quantized_vector.to_vec()
    }

    fn distance(
        &self,
        _query: &[f32],
        _point: &[f32],
        _implem: utils::distance::l2::L2DistanceCalculatorImpl,
    ) -> f32 {
        D::calculate(_query, _point)
    }

    fn read(dir: String) -> Result<Self>
    where
        Self: Sized,
    {
        let reader = NoQuantizerReader::new(dir);
        reader.read()
    }
}

impl<D: DistanceCalculator> WritableQuantizer for NoQuantizer<D> {
    fn write_to_directory(&self, base_directory: &str) -> Result<()> {
        let config = NoQuantizerConfig {
            dimension: self.dimension,
        };
        std::fs::write(
            &format!("{}/no_op_quantizer_config.yaml", base_directory),
            serde_yaml::to_string(&config)?,
        )?;
        Ok(())
    }
}

#[derive(Serialize, Deserialize)]
pub struct NoQuantizerConfig {
    pub dimension: usize,
}

pub struct NoQuantizerReader<D: DistanceCalculator> {
    base_directory: String,

    _marker: PhantomData<D>,
}

impl<D: DistanceCalculator> NoQuantizerReader<D> {
    pub fn new(base_directory: String) -> Self {
        Self {
            base_directory,
            _marker: PhantomData,
        }
    }

    pub fn read(&self) -> Result<NoQuantizer<D>> {
        // Deserialieze the config
        let config = serde_yaml::from_str::<NoQuantizerConfig>(&std::fs::read_to_string(
            &format!("{}/no_op_quantizer_config.yaml", self.base_directory),
        )?)?;
        Ok(NoQuantizer::new(config.dimension))
    }
}

pub type NoQuantizerL2 = NoQuantizer<L2DistanceCalculator>;
