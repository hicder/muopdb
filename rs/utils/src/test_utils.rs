use rand::Rng;

// Generate a random vector with a given dimension
pub fn generate_random_vector(dimension: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let mut vector = vec![];
    for _ in 0..dimension {
        vector.push(rng.gen::<f32>());
    }
    vector
}
