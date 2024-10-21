use crate::DistanceCalculator;

pub struct L2DistanceCalculator {}

impl L2DistanceCalculator {
    pub fn new() -> Self {
        Self {}
    }
}

impl DistanceCalculator for L2DistanceCalculator {
    /// Compute L2 distance between two vectors
    fn calculate(&self, a: &[f32], b: &[f32]) -> f32 {
        let mut dist = 0.0;
        for i in 0..a.len() {
            dist += (a[i] - b[i]).powi(2);
        }
        dist.sqrt()
    }
}
