use anyhow::Result;
use bit_vec::BitVec;
use ndarray::{Array1, Array2, Axis};
use ndarray_linalg::normalize;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::StandardNormal; // TODO: maybe uniform instead?
use ndarray_linalg::qr::QR;
use ndarray_linalg::norm::Norm;
use ndarray_linalg::solve::Inverse;
use std::error::Error;

pub struct RabitQBuilder {
    dimension: usize,
    dataset: Vec<f32>,
}

impl RabitQBuilder {
    pub fn new(
        dimension: usize
    ) -> Self {
        Self {
            dimension,
            dataset: Vec::new()
        }
    }

    pub fn add(&mut self, data: Vec<f32>) {
        debug_assert!(data.len() == self.dimension, "Vector must have the same dimension as the dataset");

        self.dataset.append(&mut data.clone());
    }

    pub fn build(&mut self, _base_directory: &str) -> Result<RabitQ, Box<dyn Error>> {
        assert!(self.dataset.len() > 0, "Dataset is empty");

        let q = self.create_orthogonal_matrix(self.dimension)?;

        let shape = (self.dataset.len() / self.dimension, self.dimension);
        let dataset = Array2::from_shape_vec(shape, self.dataset.clone())?;
        let centroid: Array1<f32> = dataset.mean_axis(Axis(0)).ok_or("Failed to calculate mean")?;

        let dist_from_centroid = self.get_dist_from_centroid(&centroid, &dataset);

        Ok(RabitQ {
            p_inv: q.inv()?,
            centroid,
            quantization_codes: Vec::new(),
            dist_from_centroid,
            o_o: Vec::new(),
        })
    }

    fn create_orthogonal_matrix(&self, d: usize)  -> Result<Array2<f32>, Box<dyn Error>>{
        let matrix: Array2<f32> = Array2::random((d, d), StandardNormal);
        let (q, _) = matrix.qr()?;
        return Ok(q);
    }

    fn get_dist_from_centroid(&self, centroid: &Array1<f32>, dataset: &Array2<f32>) -> Array1<f32> {
        let differences = dataset - centroid;
        let mut norm = Vec::new();
        for row in differences.axis_iter(Axis(0)) {
            norm.push(row.norm_l2());
        }

        return Array1::from_vec(norm);
    }
}

pub struct RabitQ {
    pub p_inv: Array2<f32>,
    pub centroid: Array1<f32>,
    pub quantization_codes: Vec<BitVec>,
    pub dist_from_centroid: Array1<f32>,
    pub o_o: Vec<Vec<f32>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_create_orthogonal_matrix() {
        let mut builder = RabitQBuilder::new(2);
        builder.add(vec![1.0, 2.0]);

        let rabitq = builder.build("foo").unwrap();
        let _pi: &Array2<f32> = &rabitq.p_inv;

        assert_eq!(_pi.shape(), &[2, 2]);

        // TODO: use approx in ndarray to compare (can't fiture out how it works atm)
        // Orthogonal matrix means A * A^T = I
        let identity_matrix = Array2::eye(2);
        let actual = _pi.dot(&_pi.t());
        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(actual[[i, j]], identity_matrix[[i, j]]);
            }
        }
    }

    #[test]
    fn test_get_centroid() {
        let mut builder = RabitQBuilder::new(2);
        builder.add(vec![1.0, 2.0]);
        builder.add(vec![2.0, 3.0]);

        let rabitq = builder.build("foo").unwrap();
        let centroid = &rabitq.centroid;

        assert_eq!(centroid.shape(), &[2]);
        assert_eq!(centroid, &Array1::from_vec(vec![1.5, 2.5]));
    }

    #[test]
    fn test_dist_from_centroid() {
        let mut builder = RabitQBuilder::new(2);
        builder.add(vec![0.0, 0.0]);
        builder.add(vec![0.0, 2.0]);

        let rabitq = builder.build("foo").unwrap();

        assert_eq!(rabitq.dist_from_centroid, Array1::from_vec(vec![1.0f32, 1.0]));
    }
}