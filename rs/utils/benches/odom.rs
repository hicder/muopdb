use std::fs::{self, File};

use criterion::measurement::WallTime;
use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkGroup, BenchmarkId, Criterion,
};
use rand::Rng;
use utils::on_disk_ordered_map::builder::OnDiskOrderedMapBuilder;
use utils::on_disk_ordered_map::encoder::{FixedIntegerCodec, IntegerCodec, VarintIntegerCodec};
use utils::on_disk_ordered_map::map::OnDiskOrderedMap;
use uuid::Uuid;

fn run_benchmark<C: IntegerCodec>(num_keys: usize, group: &mut BenchmarkGroup<WallTime>) {
    // Create temporary directory for the test
    let tmp_dir = tempdir::TempDir::new("bench_odom").unwrap();
    fs::create_dir_all(tmp_dir.path()).unwrap();
    let map_path = tmp_dir.path().join("test.odom");
    let map_path_str = map_path.to_str().unwrap();

    // Build the map
    let mut builder = OnDiskOrderedMapBuilder::new();
    let mut keys = Vec::with_capacity(num_keys);

    // Generate and insert keys
    for i in 0..num_keys {
        let key = Uuid::new_v4().to_string();
        builder.add(key.clone(), i as u64);
        keys.push(key);
    }

    // Build the map with FixedIntegerCodec
    builder.build(C::new(), map_path_str).unwrap();

    // Open the map for reading
    let file = File::open(&map_path).unwrap();
    let mmap = unsafe { memmap2::Mmap::map(&file) }.unwrap();
    let map = OnDiskOrderedMap::<C>::new(map_path_str.to_string(), &mmap, 0, mmap.len()).unwrap();

    let mut rng = rand::thread_rng();

    let codec = C::new();
    let codec_name = match codec.id() {
        0 => "FixedIntegerCodec",
        1 => "VarintIntegerCodec",
        _ => "UnknownCodec",
    };
    let bench_name = "Search/".to_owned() + codec_name;

    group.bench_with_input(
        BenchmarkId::new(bench_name, num_keys),
        &num_keys,
        |bencher, _| {
            bencher.iter(|| {
                let key_idx = rng.gen_range(0..keys.len());
                let key = &keys[key_idx];
                black_box(map.get(key).unwrap())
            })
        },
    );
}

fn bench_odom(c: &mut Criterion) {
    let mut group = c.benchmark_group("ODOM");

    for codec in [0, 1].iter() {
        for num_keys in [16384, 32768, 65536].iter() {
            if *codec == 0 {
                run_benchmark::<FixedIntegerCodec>(*num_keys, &mut group);
            } else {
                run_benchmark::<VarintIntegerCodec>(*num_keys, &mut group);
            }
        }
    }

    group.finish();
}

criterion_group!(benches, bench_odom);
criterion_main!(benches);
