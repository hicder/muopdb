use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

use config::collection::CollectionConfig;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use index::collection::core::Collection;
use index::collection::reader::CollectionReader;
use index::wal::entry::WalOpType;
use quantization::noq::noq::NoQuantizerL2;
use tempdir::TempDir;
use tokio::task::JoinSet;
use utils::test_utils::generate_random_vector;

fn bench_wal_insertion(c: &mut Criterion) {
    let mut group = c.benchmark_group("WalInsertion");
    // Run only 10 iterations since it's pretty slow
    group.sample_size(10);

    let collection_name = "test_collection_wal";
    let temp_dir = TempDir::new(collection_name).expect("Failed to create temporary directory");
    let base_directory: String = temp_dir.path().to_str().unwrap().to_string();
    let segment_config = CollectionConfig {
        num_features: 128,
        // Enable WAL for this benchmark with write grouping
        wal_file_size: 1024 * 1024, // 1MB WAL file size
        wal_write_group_size: 940,  // Group size for batching
        ..CollectionConfig::default()
    };

    // Remove everything under base_directory
    std::fs::remove_dir_all(&base_directory).unwrap();

    // init the collection
    Collection::<NoQuantizerL2>::init_new_collection(base_directory.clone(), &segment_config)
        .unwrap();
    let reader = CollectionReader::new(collection_name.to_string(), base_directory.clone());
    let collection = reader.read::<NoQuantizerL2>().unwrap();

    // Start background thread to process pending operations outside of benchmark
    let running = Arc::new(AtomicBool::new(true));
    let running_clone = running.clone();
    let collection_clone = collection.clone();
    let background_thread = thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().unwrap();
        while running_clone.load(Ordering::Relaxed) {
            // Process pending operations in a blocking context
            rt.block_on(async {
                loop {
                    let processed = collection_clone.process_one_op().await.unwrap();
                    if processed == 0 {
                        break;
                    }
                }
            });
            // Small delay to prevent busy looping
            thread::sleep(Duration::from_millis(1));
        }
    });

    let num_vectors = 1000;
    let num_features = segment_config.num_features;
    let vectors: Vec<Arc<[f32]>> = (0..num_vectors)
        .map(|_| Arc::from(generate_random_vector(num_features).as_slice()))
        .collect();
    let vectors = Arc::new(vectors); // Share the entire collection
    let user_ids: Arc<[u128]> = Arc::from(vec![0].as_slice());

    // Number of concurrent tasks
    const NUM_TASKS: usize = 1000;
    println!(
        "num tasks: {NUM_TASKS}, wal group size: {}",
        segment_config.wal_write_group_size,
    );
    // Use async runtime for proper concurrent context
    let rt = tokio::runtime::Runtime::new().unwrap();

    group.bench_with_input(
        BenchmarkId::new("WalInsertion", 1000),
        &1000,
        |bencher, _| {
            bencher.iter(|| {
                rt.block_on(async {
                    let mut join_set = JoinSet::new();

                    // Spawn concurrent async tasks instead of threads
                    for task_id in 0..NUM_TASKS {
                        let collection_clone = collection.clone();
                        let user_ids_clone = user_ids.clone();
                        let vectors_clone = vectors.clone(); // Just cloning the Arc pointer

                        join_set.spawn(async move {
                            let doc_id = task_id as u128;
                            let data = vectors_clone[task_id % vectors_clone.len()].clone();
                            let user_ids_ref = user_ids_clone.clone();
                            let doc_ids: Arc<[u128]> = Arc::from([doc_id]);

                            collection_clone
                                .write_to_wal(doc_ids, user_ids_ref, WalOpType::Insert(data))
                                .await
                                .unwrap()
                        });
                    }

                    // Wait for all tasks to complete concurrently
                    while let Some(result) = join_set.join_next().await {
                        result.unwrap(); // Ensure no task panicked
                    }
                });
            });
        },
    );

    // Stop the background threads
    running.store(false, Ordering::Relaxed);
    background_thread.join().unwrap();

    group.finish();
}

criterion_group!(benches, bench_wal_insertion);
criterion_main!(benches);
