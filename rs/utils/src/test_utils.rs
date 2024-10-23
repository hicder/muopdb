use rand::Rng;

// Generate a random vector with a given dimension
pub fn generate_random_vector(dimension: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..dimension).map(|_| rng.gen()).collect()
}

// This test is used to generate 10000 vectors of dimension 128
#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;

    #[test]
    fn test_generate_random_vector() {
        let dimension = 128;
        let mut results = vec![];
        for _ in 0..10000 {
            results.push(generate_random_vector(dimension));
        }

        // Write to /tmp/dataset.bin
        let mut file = std::fs::File::create("/tmp/dataset.bin").unwrap();
        for result in results {
            for i in 0..dimension {
                file.write_all(&result[i].to_le_bytes()).unwrap();
            }
        }
        file.flush().unwrap();
    }
}
