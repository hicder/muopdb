
use criterion::{criterion_group, criterion_main, black_box, Criterion};
use utils::{distance::dot_product::DotProductDistanceCalculator, test_utils::generate_random_vector, DistanceCalculator};

fn benches_dot_product(c: &mut Criterion) {
    let a = generate_random_vector(128);
    let b = generate_random_vector(128);
    let mut group = c.benchmark_group("dot_product");
    group.bench_function("benchmark-scalar", |bench| {
        bench.iter(|| {
            DotProductDistanceCalculator::calculate_scalar(black_box(&a), black_box(&b));
        });
    });

    group.bench_function("benchmark-simd", |bench| {
        bench.iter(|| {
            DotProductDistanceCalculator::calculate(black_box(&a), black_box(&b));
        });
    });
    group.finish();
}

criterion_group!(benches, benches_dot_product);
criterion_main!(benches);