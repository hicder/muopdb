use crate::DistanceCalculator;

pub struct L2DistanceCalculator {}

impl DistanceCalculator for L2DistanceCalculator {
    fn calculate(&self, a: &[f32], b: &[f32]) -> f32 {
        let mut dist = 0.0;
        for i in 0..a.len() {
            dist += (a[i] - b[i]).powi(2);
        }
        dist.sqrt()
    }
}
