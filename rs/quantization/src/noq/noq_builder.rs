use anyhow::Result;

use crate::noq::noq::{NoQuantizer, NoQuantizerConfig};

pub struct NoQuantizerBuilder {
    config: NoQuantizerConfig,
}

impl NoQuantizerBuilder {
    /// Create a new NoQuantizerBuilder
    pub fn new(config: NoQuantizerConfig) -> Self {
        Self { config }
    }

    pub fn build(&mut self) -> Result<NoQuantizer> {
        Ok(NoQuantizer::new(self.config.dimension))
    }
}
