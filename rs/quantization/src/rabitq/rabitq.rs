use bit_vec::BitVec;
use ndarray::{Array1, Array2};

pub struct RabitQ {
    // The centroid of all the data points
    // Notation: $c$
    // Dimension: $D$
    pub centroid: Array1<f32>,

    // The inverse of the orthogonal matrix P
    // Notation: $P^{-1}$
    // Dimension: $D \times D$
    pub orthogonal_matrix_inv: Array2<f32>,

    // The quantization code for each data point
    // Notation: $\bar{x}_b$
    // Dimension: $N \times D$
    pub quantization_codes: Vec<BitVec>,

    // The distance of each data point from the centroid
    // Notation: $||o_r - c||$
    // Dimension: $N$
    pub dist_from_centroid: Vec<f32>,

    // The dot products of each quantized vector with its data point
    // Notation: $<\bar{o}, o>$
    // Dimension: $N$
    pub quantized_vector_dot_products: Array1<f32>,
}
