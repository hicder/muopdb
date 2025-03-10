use utils::distance::l2::L2DistanceCalculator;
use utils::kmeans_builder::kmeans_builder::{KMeansBuilder, KMeansVariant};

fn main() {
    env_logger::init();
    let dimension = 128;
    let num_datapoints = 1000000;

    let mut flattened_dataset = vec![0.0; dimension * num_datapoints];
    for i in 0..num_datapoints {
        for j in 0..dimension {
            flattened_dataset[i * dimension + j] = i as f32;
        }
    }

    let kmeans =
        KMeansBuilder::<L2DistanceCalculator>::new(10000, 5, 0.0, dimension, KMeansVariant::Lloyd);
    let _result = kmeans
        .fit(flattened_dataset)
        .expect("Failed to run KMeans model");
}
