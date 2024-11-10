use anyhow::Result;
use bit_vec::BitVec;
use ndarray::Array2;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_linalg::qr::QR;
use ndarray_linalg::solve::Inverse;
use std::error::Error;

pub struct RabitQBuilder {
    dimension: usize,
    dataset: Vec<Vec<f32>>,
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
        self.dataset.push(data);

    }

    pub fn build(&mut self, _base_directory: String) -> Result<RabitQ, Box<dyn Error>> {
        let q = self.create_orthogonal_matrix(self.dimension)?;

        Ok(RabitQ {
            p_inv: q.inv()?,
            quantization_codes: Vec::new(),
            or_c: Vec::new(),
            o_o: Vec::new(),
        })
    }

    pub fn create_orthogonal_matrix(&self, d: usize)  -> Result<Array2<f32>, Box<dyn Error>>{
        let matrix: Array2<f32> = Array2::random((d, d), StandardNormal);
        let (q, _) = matrix.qr()?;
        return Ok(q);
    }
}

pub struct RabitQ {
    p_inv: Array2<f32>,
    quantization_codes: Vec<BitVec>,
    or_c: Vec<f32>,
    o_o: Vec<Vec<f32>>,
}