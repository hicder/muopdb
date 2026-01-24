use anyhow::Result;
use bit_vec::BitVec;
use ndarray::{Array1, Array2, Axis};
use ndarray_linalg::norm::{normalize, NormalizeAxis};
use ndarray_linalg::qr::QR;
use ndarray_linalg::solve::Inverse;
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;

use crate::rabitq::RabitQ;

pub struct RabitQBuilder {
    dimension: usize,
    dataset: Vec<f32>, // size = sample_count * dimension
}

impl RabitQBuilder {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            dataset: Vec::new(),
        }
    }

    pub fn add(&mut self, data: Vec<f32>) {
        debug_assert!(
            data.len() == self.dimension,
            "Vector must have the same dimension as the dataset"
        );

        self.dataset.append(&mut data.clone());
    }

    pub fn build(&mut self) -> Result<RabitQ> {
        assert!(!self.dataset.is_empty(), "Dataset is empty");

        let dataset = self.get_dataset()?;

        // 1. Normalize the set of vectors and store ||o_r - c||
        // Note the paper suggests using multiple centroids, but we'll start with one for simplicity
        let centroid: Array1<f32> = self.get_centroid(&dataset);
        let (normalized_dataset, dist_from_centroid) =
            normalize(&dataset - &centroid, NormalizeAxis::Row);

        // 2. Sample a random orthogonal matrix P to construct the codebook C_rand
        let p = self.generate_orthogonal_matrix(self.dimension)?;
        let p_inv = p.inv()?;

        // 3. Compute the quantization code x_b
        let quantization_codes = self.get_quantization_codes(&normalized_dataset, &p_inv);

        // 4. Pre-compute the values of <\bar{o}, o>
        let quantized_vector_dot_products =
            self.get_quantized_vector_dot_products(&normalized_dataset, &quantization_codes, &p);

        Ok(RabitQ {
            centroid,
            orthogonal_matrix_inv: p_inv,
            quantization_codes,
            dist_from_centroid,
            quantized_vector_dot_products,
        })
    }

    fn get_dataset(&self) -> Result<Array2<f32>> {
        let sample_count: usize = self.dataset.len() / self.dimension;
        Ok(Array2::from_shape_vec(
            (sample_count, self.dimension),
            self.dataset.clone(),
        )?)
    }

    fn get_centroid(&self, dataset: &Array2<f32>) -> Array1<f32> {
        dataset.mean_axis(Axis(0)).unwrap()
    }

    fn generate_orthogonal_matrix(&self, d: usize) -> Result<Array2<f32>> {
        let matrix: Array2<f32> = Array2::random((d, d), StandardNormal);
        let (q, _) = matrix.qr()?;
        Ok(q)
    }

    fn get_quantization_codes(
        &self,
        normalized_dataset: &Array2<f32>,
        p_inv: &Array2<f32>,
    ) -> Vec<BitVec> {
        // Each data point is quantized as
        // x_b = sign(P^{−1}o)
        normalized_dataset
            .axis_iter(Axis(0))
            .map(|datapoint| {
                p_inv
                    .dot(&datapoint)
                    .map(|x| *x > 0.0)
                    .into_iter()
                    .collect::<BitVec>()
            })
            .collect()
    }

    fn get_quantized_vector_dot_products(
        &self,
        normalized_dataset: &Array2<f32>,
        quantization_codes: &[BitVec],
        p: &Array2<f32>,
    ) -> Array1<f32> {
        debug_assert!(
            quantization_codes.len() == normalized_dataset.len_of(Axis(0)),
            "The number of quantization codes {} must be equal to the number of data points {}",
            quantization_codes.len(),
            normalized_dataset.len_of(Axis(0))
        );

        let sphere_coordinate: f32 = 1.0 / (self.dimension as f32).sqrt(); // coordinate of the unit sphere in D-dimensional space
        quantization_codes
            .iter()
            .zip(normalized_dataset.axis_iter(Axis(0)))
            .map(|(code, datapoint)| {
                let x_bar: Array1<f32> = code
                    .iter()
                    .map(|x| {
                        if x {
                            sphere_coordinate
                        } else {
                            -sphere_coordinate
                        }
                    })
                    .collect();
                let quantized_vector: Array1<f32> = p.dot(&x_bar);
                quantized_vector.dot(&datapoint)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;

    const EPSILON: f32 = 0.0001;

    #[test]
    fn test_create_orthogonal_matrix() {
        let mut builder = RabitQBuilder::new(2);
        builder.add(vec![1.0, 2.0]);

        let rabitq = builder.build().unwrap();
        let _pi: &Array2<f32> = &rabitq.orthogonal_matrix_inv;

        assert_eq!(_pi.shape(), &[2, 2]);

        // TODO: use approx in ndarray to compare (can't fiture out how it works atm)
        // Orthogonal matrix means A * A^T = I
        let identity_matrix = Array2::eye(2);
        let actual = _pi.dot(&_pi.t());
        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(actual[[i, j]], identity_matrix[[i, j]], epsilon = EPSILON);
            }
        }
    }

    #[test]
    fn test_get_centroid() {
        let mut builder = RabitQBuilder::new(2);
        builder.add(vec![1.0, 2.0]);
        builder.add(vec![2.0, 3.0]);

        let rabitq = builder.build().unwrap();
        let centroid = &rabitq.centroid;

        assert_eq!(centroid.shape(), &[2]);
        assert_abs_diff_eq!(centroid[0], 1.5, epsilon = EPSILON);
        assert_abs_diff_eq!(centroid[1], 2.5, epsilon = EPSILON);
    }

    #[test]
    fn test_dist_from_centroid() {
        let mut builder = RabitQBuilder::new(2);
        builder.add(vec![0.0, 0.0]);
        builder.add(vec![0.0, 2.0]);

        let rabitq = builder.build().unwrap();

        assert_abs_diff_eq!(rabitq.dist_from_centroid[0], 1.0, epsilon = EPSILON);
        assert_abs_diff_eq!(rabitq.dist_from_centroid[1], 1.0, epsilon = EPSILON);
    }

    #[test]
    fn test_get_quantization_code() {
        let dimension = 4;

        let mut builder = RabitQBuilder::new(dimension);
        builder.add(vec![0.0; dimension]);
        builder.add(vec![2.0; dimension]);

        let rabitq = builder.build().unwrap();

        assert_eq!(rabitq.quantization_codes.len(), 2);
        assert_eq!(rabitq.quantization_codes[0].len(), dimension);
        assert_eq!(rabitq.quantization_codes[1].len(), dimension);
    }

    #[test]
    fn test_get_quantized_vector_dot_products() {
        let dimension = 4;

        let mut builder = RabitQBuilder::new(dimension);
        builder.add(vec![0.0; dimension]);
        builder.add(vec![1.0; dimension]);

        let rabitq = builder.build().unwrap();

        assert_eq!(rabitq.quantized_vector_dot_products.shape(), &[2]);
    }
}
