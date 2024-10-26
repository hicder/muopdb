use criterion::{black_box, criterion_group, criterion_main, Criterion};
use quantization::pq::ProductQuantizerConfig;
use quantization::pq_builder::{ProductQuantizerBuilder, ProductQuantizerBuilderConfig};
use quantization::quantization::Quantizer;
use utils::test_utils::generate_random_vector;

fn bench_pq_distance(c: &mut Criterion) {
    for dimension in [128, 256].iter() {
        for subvector_dimension in [4, 8, 16, 32, 64, 128].iter() {
            for num_bits in [4, 8, 16].iter() {
                let mut pqb = ProductQuantizerBuilder::new(
                    ProductQuantizerConfig {
                        dimension: *dimension,
                        subvector_dimension: *subvector_dimension,
                        num_bits: *num_bits,
                        base_directory: "bm".to_string(),
                        codebook_name: "bm".to_string(),
                    },
                    ProductQuantizerBuilderConfig {
                        max_iteration: 1000,
                        batch_size: 4,
                    },
                );
                let sample_size = 1 << *num_bits;
                for _ in 0..sample_size {
                    pqb.add(generate_random_vector(*dimension));
                }

                let pq = pqb.build().unwrap();
                let point = pq.quantize(&generate_random_vector(*dimension));
                let query = pq.quantize(&generate_random_vector(*dimension));
                c.bench_function(
                    &format!(
                        "pq_distance_scalar_{}_{}_{}",
                        *dimension, *subvector_dimension, *num_bits
                    ),
                    |b| b.iter(|| pq.distance(black_box(&query), black_box(&point))),
                );
            }
        }
    }
}

criterion_group!(benches, bench_pq_distance);
criterion_main!(benches);
