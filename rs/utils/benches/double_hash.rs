use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use utils::bloom_filter::bloom_filter::InMemoryBloomFilter;

fn bench_double_hash(c: &mut Criterion) {
    let mut group = c.benchmark_group("double_hash_methods");

    // Create bloom filters with different sizes
    let filter_sizes = [1_000, 10_000, 100_000];
    let false_positive_rate = 0.01;

    // Test subjects for hashing - use various types that don't have private fields
    let string_value = "benchmark_string".to_string();
    let u64_value = 12345678u64;
    let tuple_value = (987654321u128, 123456789u128);

    // Test with different seeds
    let seeds = [0u64, 1u64, 42u64, 9999u64];

    for &size in &filter_sizes {
        // Create bloom filters for each type
        let string_filter = InMemoryBloomFilter::<String>::new(size, false_positive_rate);
        let u64_filter = InMemoryBloomFilter::<u64>::new(size, false_positive_rate);
        let tuple_filter = InMemoryBloomFilter::<(u128, u128)>::new(size, false_positive_rate);

        // Benchmark each function for String type
        for &seed in &seeds {
            // Include the seed in the benchmark ID to make it unique
            group.bench_with_input(
                BenchmarkId::new(format!("double_hash_string_seed{}", seed), size),
                &size,
                |b, _| {
                    b.iter(|| {
                        black_box(
                            string_filter.double_hash(black_box(&string_value), black_box(seed)),
                        )
                    })
                },
            );

            group.bench_with_input(
                BenchmarkId::new(format!("double_hash_alt_string_seed{}", seed), size),
                &size,
                |b, _| {
                    b.iter(|| {
                        black_box(
                            string_filter
                                .double_hash_alt(black_box(&string_value), black_box(seed)),
                        )
                    })
                },
            );
        }

        // Benchmark each function for u64 type
        for &seed in &seeds {
            group.bench_with_input(
                BenchmarkId::new(format!("double_hash_u64_seed{}", seed), size),
                &size,
                |b, _| {
                    b.iter(|| {
                        black_box(u64_filter.double_hash(black_box(&u64_value), black_box(seed)))
                    })
                },
            );

            group.bench_with_input(
                BenchmarkId::new(format!("double_hash_alt_u64_seed{}", seed), size),
                &size,
                |b, _| {
                    b.iter(|| {
                        black_box(
                            u64_filter.double_hash_alt(black_box(&u64_value), black_box(seed)),
                        )
                    })
                },
            );
        }

        // Benchmark each function for tuple type (instead of HashKey with private fields)
        for &seed in &seeds {
            group.bench_with_input(
                BenchmarkId::new(format!("double_hash_tuple_seed{}", seed), size),
                &size,
                |b, _| {
                    b.iter(|| {
                        black_box(
                            tuple_filter.double_hash(black_box(&tuple_value), black_box(seed)),
                        )
                    })
                },
            );

            group.bench_with_input(
                BenchmarkId::new(format!("double_hash_alt_tuple_seed{}", seed), size),
                &size,
                |b, _| {
                    b.iter(|| {
                        black_box(
                            tuple_filter.double_hash_alt(black_box(&tuple_value), black_box(seed)),
                        )
                    })
                },
            );
        }
    }

    group.finish();
}

// Benchmark insert and lookup operations which use the hash functions internally
fn bench_bloom_filter_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("bloom_filter_access_patterns");

    // Test with a moderately-sized bloom filter
    let expected_elements = 100_000;
    let false_positive_rate = 0.01;

    // Create test values using tuples instead of HashKey to avoid private fields
    let test_values: Vec<(u128, u128)> = (0..100).map(|i| (i as u128, (i * 10) as u128)).collect();

    // Benchmark using standard insert (which uses double_hash internally)
    let mut filter =
        InMemoryBloomFilter::<(u128, u128)>::new(expected_elements, false_positive_rate);

    group.bench_function("standard_insert_lookup", |b| {
        // Reset the filter before benchmarking
        filter.clear();

        b.iter(|| {
            // Insert values
            for value in &test_values {
                filter.insert(black_box(value));
            }

            // Check if values exist
            for value in &test_values {
                black_box(filter.may_contain(black_box(value)));
            }
        })
    });

    // Create a custom benchmark that directly calls the hash functions
    group.bench_function("direct_hash_calculation", |b| {
        let filter =
            InMemoryBloomFilter::<(u128, u128)>::new(expected_elements, false_positive_rate);

        // Just benchmark the direct calculation of hash values
        b.iter(|| {
            for value in &test_values {
                for i in 0..10 {
                    // Use a fixed number of iterations
                    black_box(filter.double_hash(black_box(value), black_box(i)));
                }
            }
        })
    });

    group.bench_function("direct_hash_alt_calculation", |b| {
        let filter =
            InMemoryBloomFilter::<(u128, u128)>::new(expected_elements, false_positive_rate);

        // Benchmark the alternative hash calculation
        b.iter(|| {
            for value in &test_values {
                for i in 0..10 {
                    // Use a fixed number of iterations
                    black_box(filter.double_hash_alt(black_box(value), black_box(i)));
                }
            }
        })
    });

    group.finish();
}

criterion_group!(benches, bench_double_hash, bench_bloom_filter_operations);
criterion_main!(benches);
