use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use utils::distance::dot_product_similarity::DotProductSimilarityCalculator;
use utils::test_utils::generate_random_vector;
use utils::DistanceCalculator;

fn bench_dot_product_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("Dot Product Similarity");
    let mut distance_calculator = DotProductSimilarityCalculator::new();

    for size in [
        8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096,
        384,  // VECTOR_DIM_SENTENCE_TRANSFORMERS_MINI_LM
        768,  // VECTOR_DIM_SENTENCE_TRANSFORMERS_MPNET
        1536, // VECTOR_DIM_OPENAI_SMALL
        3072, // VECTOR_DIM_OPENAI_LARGE
    ].iter()
    {
        let a = generate_random_vector(*size);
        let b = generate_random_vector(*size);

        group.bench_with_input(BenchmarkId::new("Scalar", *size), &size, |bencher,_| {
            bencher.iter(|| distance_calculator.calculate_scalar(black_box(&a), black_box(&b)))
        });

        group.bench_with_input(BenchmarkId::new("SIMD", *size), &size, |bencher,_| {
            bencher.iter(|| distance_calculator.calculate_simd(black_box(&a), black_box(&b)))
        });

        group.bench_with_input(BenchmarkId::new("Calculate", *size), &size, |bencher,_| {
            bencher.iter(|| distance_calculator.calculate(black_box(&a), black_box(&b)))
        });
    }

    group.finish();
}

criterion_group!(benches, bench_dot_product_similarity);
criterion_main!(benches);