use std::marker::PhantomData;

use anyhow::Result;
use utils::DistanceCalculator;

use crate::noq::noq::{NoQuantizer, NoQuantizerConfig};

pub struct NoQuantizerBuilder<D: DistanceCalculator> {
    config: NoQuantizerConfig,

    _marker: PhantomData<D>,
}

impl<D: DistanceCalculator> NoQuantizerBuilder<D> {
    /// Create a new NoQuantizerBuilder
    pub fn new(config: NoQuantizerConfig) -> Self {
        Self {
            config,
            _marker: PhantomData,
        }
    }

    pub fn build(&mut self) -> Result<NoQuantizer<D>> {
        Ok(NoQuantizer::new(self.config.dimension))
    }
}
