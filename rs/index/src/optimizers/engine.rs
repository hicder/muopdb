use std::sync::Arc;

use anyhow::Result;

use super::noop::NoopOptimizer;
use crate::collection::Collection;

#[derive(Debug, PartialEq, Eq)]
pub enum OptimizingType {
    Vacuum,
    Merge,
    Noop,
}

pub struct OptimizerEngine {
    collection: Arc<Collection>,
}

impl OptimizerEngine {
    pub fn new(collection: Arc<Collection>) -> Self {
        Self { collection }
    }

    pub fn run(&self, segments: Vec<String>, optimizing_type: OptimizingType) -> Result<()> {
        if segments.len() < 2 && optimizing_type == OptimizingType::Merge {
            return Ok(());
        }

        let pending_segment = self.collection.init_optimizing(&segments)?;
        match optimizing_type {
            OptimizingType::Vacuum => Ok(()),
            OptimizingType::Merge => Ok(()),
            OptimizingType::Noop => {
                let noop_optimizer = NoopOptimizer::new();
                self.collection
                    .run_optimizer(&noop_optimizer, &pending_segment)?;
                Ok(())
            }
        }
    }
}
