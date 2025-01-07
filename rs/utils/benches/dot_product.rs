use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use utils::distance::dot_product::DotProductDistanceCalculator;
use utils::test_utils::generate_random_vector;
use utils::DistanceCalculator;

fn benches_dot_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_product");

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

        group.bench_with_input(BenchmarkId::new("simd", size), &size, |bench, &_size| {
            bench.iter(|| {
                DotProductDistanceCalculator::calculate(black_box(&a), black_box(&b));
            });
        });

        group.bench_with_input(BenchmarkId::new("scalar", size), &size, |bench, &_size| {
            bench.iter(|| {
                DotProductDistanceCalculator::calculate_scalar(black_box(&a), black_box(&b));
            });
        });
    }
    group.finish();
}

criterion_group!(benches, benches_dot_product);
criterion_main!(benches);
