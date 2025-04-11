use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rkyv::util::AlignedVec;
use utils::mem::{transmute_u8_to_val_aligned, transmute_u8_to_val_unaligned};
use utils::test_utils::generate_random_vector_generic;

fn bench_mem(c: &mut Criterion) {
    let mut group = c.benchmark_group("Mem Read");
    let b_unaligned = generate_random_vector_generic::<u8>(17);
    let mut b_aligned = AlignedVec::<16>::with_capacity(17);
    for v in b_unaligned.iter() {
        b_aligned.push(*v);
    }

    group.bench_with_input(BenchmarkId::new("Unalign", false), &false, |bencher, _| {
        bencher.iter(|| transmute_u8_to_val_unaligned::<u64>(&b_unaligned))
    });

    group.bench_with_input(BenchmarkId::new("Align", true), &true, |bencher, _| {
        bencher.iter(|| transmute_u8_to_val_aligned::<u64>(&b_aligned))
    });
    group.finish();
}

criterion_group!(benches, bench_mem);
criterion_main!(benches);
