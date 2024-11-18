use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use utils::distance::l2::L2DistanceCalculator;
use utils::test_utils::generate_random_vector;
use utils::DistanceCalculator;

fn bench_l2_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("L2 Distance");
    for size in [
        8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096,
        384,  // VECTOR_DIM_SENTENCE_TRANSFORMERS_MINI_LM
        768,  // VECTOR_DIM_SENTENCE_TRANSFORMERS_MPNET
        1536, // VECTOR_DIM_OPENAI_SMALL
        3072, // VECTOR_DIM_OPENAI_LARGE
    ]
    .iter()
    {
        let a = generate_random_vector(*size);
        let b = generate_random_vector(*size);

        group.bench_with_input(BenchmarkId::new("Scalar", *size), &size, |bencher, _| {
            bencher.iter(|| L2DistanceCalculator::calculate_scalar(black_box(&a), black_box(&b)))
        });

        group.bench_with_input(BenchmarkId::new("SIMD", *size), &size, |bencher, _| {
            bencher.iter(|| L2DistanceCalculator::calculate(black_box(&a), black_box(&b)))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_l2_distance);
criterion_main!(benches);
