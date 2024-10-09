use pq::ProductQuantizer;

mod pq;

fn main() {
    let pq = ProductQuantizer{
        dimension: 10,
        subspace_dimension: 10,
        num_bits: 8,
        codebook: vec![0.0; 10],    
    };
    println!("Hello, world!");
}
