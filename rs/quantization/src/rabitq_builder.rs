use anyhow::Result;
use bit_vec::BitVec;
use ndarray::{Array1, Array2, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_linalg::qr::QR;
use ndarray_linalg::norm::Norm;
use ndarray_linalg::solve::Inverse;

use crate::rabitq::RabitQ;

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

    pub fn build(&mut self) -> Result<RabitQ> {
        assert!(self.dataset.len() > 0, "Dataset is empty");

        let dataset = self.get_dataset()?;
        let p = self.generate_orthogonal_matrix(self.dimension)?;
        let p_inv = p.inv()?;
        let centroid: Array1<f32> = self.get_centroid(&dataset);
        let dist_from_centroid = self.get_dist_from_centroid(&dataset, &centroid);
        let quantization_codes = self.get_quantization_codes(&dataset, &p_inv);
        let quantized_vector_dot_products = self.get_quantized_vector_dot_products(
            &dataset,
            &quantization_codes,
            &p);

        Ok(RabitQ {
            orthogonal_matrix_inv: p_inv,
            centroid,
            dist_from_centroid,
            quantization_codes,
            quantized_vector_dot_products,
        })
    }

    fn get_dataset(&self) -> Result<Array2<f32>> {
        let sample_count: usize = self.dataset.len() / self.dimension;
        Ok(Array2::from_shape_vec((sample_count, self.dimension), self.dataset.clone())?)
    }

    fn generate_orthogonal_matrix(&self, d: usize)  -> Result<Array2<f32>>{
        let matrix: Array2<f32> = Array2::random((d, d), StandardNormal);
        let (q, _) = matrix.qr()?;
        return Ok(q);
    }

    fn get_centroid(&self, dataset: &Array2<f32>) -> Array1<f32> {
        return dataset.mean_axis(Axis(0)).unwrap();
    }

    fn get_dist_from_centroid(&self, dataset: &Array2<f32>, centroid: &Array1<f32>) -> Array1<f32> {
        let differences = dataset - centroid;
        let mut norm = Vec::new();
        for row in differences.axis_iter(Axis(0)) {
            norm.push(row.norm_l2());
        }

        return Array1::from_vec(norm);
    }

    fn get_quantization_codes(&self, dataset: &Array2<f32>, p_inv: &Array2<f32>) -> Vec<BitVec>{
        let mut quantization_codes: Vec<BitVec> = Vec::new();

        // Each data point is quantized as
        // sign(P^{−1}o) , where `sign` is 1 if the element is positive and 0 otherwise.
        for row in dataset.axis_iter(Axis(0)) {
            let mut code = BitVec::with_capacity(self.dimension);
            for x in p_inv.dot(&row).iter() {
                code.push(if *x > 0.0 { true } else { false });
            }
            quantization_codes.push(code);
        }

        return quantization_codes;
    }

    fn get_quantized_vector_dot_products(&self, dataset: &Array2<f32>, quantization_codes: &Vec<BitVec>, p: &Array2<f32>) -> Array1<f32> {
        debug_assert!(
            quantization_codes.len() == dataset.len_of(Axis(0)),
            "The number of quantization codes {} must be equal to the number of data points {}",
            quantization_codes.len(),
            dataset.len_of(Axis(0)));

        // TODO: vectorize this calculation
        let positive_value = 1.0 / (self.dimension as f32).sqrt();
        let negative_value = -1.0 / (self.dimension as f32).sqrt();

        let mut dot_products: Vec<f32> = Vec::new();
        for (data, code) in dataset.axis_iter(Axis(0)).zip(quantization_codes.iter()) {
            let x_bar: Vec<f32> = code.iter()
                    .map(|x| if x { positive_value } else { negative_value })
                    .collect();
            let quantized: Array1<f32> = p.dot(&Array1::from_vec(x_bar));
            dot_products.push(quantized.dot(&data));
        }

        return Array1::from_vec(dot_products);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    const EPSILON: f32 = 0.0001;

    #[test]
    fn test_create_orthogonal_matrix() {
        let mut builder = RabitQBuilder::new(2);
        builder.add(vec![1.0, 2.0]);

        let rabitq = builder.build().unwrap();
        let _pi: &Array2<f32> = &rabitq.p_inv;

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