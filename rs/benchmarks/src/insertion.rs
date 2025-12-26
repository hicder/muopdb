use config::collection::CollectionConfig;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use index::collection::core::Collection;
use index::collection::reader::CollectionReader;
use proto::muopdb::DocumentAttribute;
use quantization::noq::noq::NoQuantizerL2;
use tempdir::TempDir;
use utils::block_cache::BlockCacheConfig;
use utils::test_utils::generate_random_vector;

fn bench_insertion(c: &mut Criterion) {
    let mut group = c.benchmark_group("Insertion");
    // Run only 10 iterations since it's pretty slow
    group.sample_size(10);

    let collection_name = "test_collection_inval";
    let temp_dir = TempDir::new(collection_name).expect("Failed to create temporary directory");
    let base_directory: String = temp_dir.path().to_str().unwrap().to_string();
    let mut segment_config = CollectionConfig::default_test_config();
    segment_config.num_features = 128;
    segment_config.wal_file_size = 0;

    let num_vectors = 10000;
    let num_features = segment_config.num_features;
    let vectors = (0..num_vectors)
        .map(|_| generate_random_vector(num_features))
        .collect::<Vec<_>>();
    let user_ids = vec![0];

    group.bench_with_input(
        BenchmarkId::new("Insertion", 10000),
        &10000,
        |bencher, _| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            bencher.iter_with_setup(
                || {
                    // Remove everything under base_directory
                    if std::path::Path::new(&base_directory).exists() {
                        std::fs::remove_dir_all(&base_directory).unwrap();
                    }

                    // init the collection
                    Collection::<NoQuantizerL2>::init_new_collection(
                        base_directory.clone(),
                        &segment_config,
                    )
                    .unwrap();
                    let reader = CollectionReader::new(
                        collection_name.to_string(),
                        base_directory.clone(),
                        Some(BlockCacheConfig::default()),
                    );
                    rt.block_on(reader.read::<NoQuantizerL2>()).unwrap()
                },
                |collection| {
                    let mut doc_id = 0;
                    rt.block_on(async {
                        for vector in vectors.iter() {
                            collection
                                .insert_for_users(
                                    &user_ids,
                                    doc_id,
                                    vector,
                                    0,
                                    DocumentAttribute::default(),
                                )
                                .await
                                .unwrap();
                            doc_id += 1;
                        }
                        collection.flush().await.unwrap();
                    });
                },
            );
        },
    );

    group.finish();
}

criterion_group!(benches, bench_insertion);
criterion_main!(benches);
