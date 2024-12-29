use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use utils::distance::l2::L2DistanceCalculator;
use utils::kmeans_builder::kmeans_builder;

fn bench_kmeans(c: &mut Criterion) {
    let mut group = c.benchmark_group("K-Means");
    let dimension = 128;
    let num_datapoints = 10000;
    let mut flattened_dataset = vec![0.0; dimension * num_datapoints];
    for i in 0..num_datapoints {
        for j in 0..dimension {
            flattened_dataset[i * dimension + j] = i as f32;
        }
    }

    let kmeans = kmeans_builder::KMeansBuilder::<L2DistanceCalculator>::new(
        100,
        1000,
        0.0,
        dimension,
        kmeans_builder::KMeansVariant::Lloyd,
    );
    for parallel in [false].iter() {
        group.bench_with_input(
            BenchmarkId::new("kmeans", parallel),
            &parallel,
            |bencher, _| {
                bencher.iter(|| {
                    if !parallel {
                        let _ = black_box(kmeans.fit(flattened_dataset.clone()));
                    } else {
                        let _ = black_box(kmeans.fit_old(flattened_dataset.clone()));
                    }
                })
            },
        );
    }
}

criterion_group!(benches, bench_kmeans);
criterion_main!(benches);
