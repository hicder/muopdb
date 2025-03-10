use std::sync::Arc;

use anyhow::{Ok, Result};
use quantization::quantization::Quantizer;

use super::vacuum::VacuumOptimizer;
use super::merge::MergeOptimizer;
use super::noop::NoopOptimizer;
use crate::collection::collection::Collection;

#[derive(Debug, PartialEq, Eq)]
pub enum OptimizingType {
    Vacuum,
    Merge,
    Noop,
}

pub struct OptimizerEngine<Q: Quantizer + Clone + Send + Sync + 'static> {
    collection: Arc<Collection<Q>>,
}

impl<Q: Quantizer + Clone + Send + Sync + 'static> OptimizerEngine<Q> {
    pub fn new(collection: Arc<Collection<Q>>) -> Self {
        Self { collection }
    }

    pub fn run(&self, segments: Vec<String>, optimizing_type: OptimizingType) -> Result<String> {
        if segments.len() < 2 && optimizing_type == OptimizingType::Merge {
            return Ok("".to_string());
        }
        if segments.len() > 1 && optimizing_type == OptimizingType::Vacuum {
            return Ok("".to_string());
        }

        let pending_segment = self.collection.init_optimizing(&segments)?;
        match optimizing_type {
            OptimizingType::Vacuum => {
                let vacuum_optimizer = VacuumOptimizer::<Q>::new();
                let new_segment_name = self
                    .collection
                    .run_optimizer(&vacuum_optimizer, &pending_segment)?;
                Ok(new_segment_name)
            }
            OptimizingType::Merge => {
                let merge_optimizer = MergeOptimizer::<Q>::new();
                let new_segment_name = self
                    .collection
                    .run_optimizer(&merge_optimizer, &pending_segment)?;
                Ok(new_segment_name)
            }
            OptimizingType::Noop => {
                let noop_optimizer = NoopOptimizer::new();
                let new_segment_name = self
                    .collection
                    .run_optimizer(&noop_optimizer, &pending_segment)?;
                Ok(new_segment_name)
            }
        }
    }
}
