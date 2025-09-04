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
        // Enable WAL for this benchmark
        wal_file_size: 1024 * 1024, // 1MB WAL file size
        ..CollectionConfig::default()
    };
    // Create the collection outside of the benchmark
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

    // Start background thread to sync WAL outside of benchmark
    let sync_running = Arc::new(AtomicBool::new(true));
    let sync_running_clone = sync_running.clone();
    let collection_clone_for_sync = collection.clone();
    let sync_thread = thread::spawn(move || {
        while sync_running_clone.load(Ordering::Relaxed) {
            // Sync WAL in a blocking context
            if let Err(e) = collection_clone_for_sync.sync_wal() {
                eprintln!("Error syncing WAL: {e}");
            }
            // Small delay to prevent busy looping
            thread::sleep(Duration::from_micros(100));
        }
    });

    let num_vectors = 1000;
    let num_features = segment_config.num_features;
    let vectors = (0..num_vectors)
        .map(|_| generate_random_vector(num_features))
        .collect::<Vec<_>>();
    let user_ids = vec![0];

    group.bench_with_input(
        BenchmarkId::new("WalInsertion", 1000),
        &1000,
        |bencher, _| {
            bencher.iter(|| {
                // Create a vector to store thread handles
                let mut handles = vec![];

                // Number of threads and documents per thread
                const NUM_THREADS: usize = 40;
                const DOCS_PER_THREAD: usize = 25;

                // Clone the collection for each thread
                let collection_clones: Vec<_> =
                    (0..NUM_THREADS).map(|_| collection.clone()).collect();

                // Spawn 10 threads, each handling 100 document IDs
                (0..NUM_THREADS).for_each(|thread_idx| {
                    let collection_clone = collection_clones[thread_idx].clone();
                    let user_ids_clone = user_ids.clone();
                    let vectors_clone = vectors.clone();

                    let handle = std::thread::spawn(move || {
                        // Create a Tokio runtime for each thread
                        let rt = tokio::runtime::Runtime::new().unwrap();
                        let start_doc_id = thread_idx * DOCS_PER_THREAD;
                        let end_doc_id = start_doc_id + DOCS_PER_THREAD;

                        (start_doc_id..end_doc_id).for_each(|doc_id| {
                            let data: Arc<[f32]> = Arc::from(vectors_clone[doc_id].as_slice());
                            let user_ids: Arc<[u128]> = Arc::from(user_ids_clone.as_slice());
                            let doc_ids: Arc<[u128]> = Arc::from([doc_id as u128]);
                            // Use write_to_wal instead of insert_for_users
                            // This is the only operation being measured in the benchmark
                            rt.block_on(collection_clone.write_to_wal(
                                doc_ids,
                                user_ids,
                                data,
                                WalOpType::Insert,
                            ))
                            .unwrap();
                        });
                    });

                    handles.push(handle);
                });

                // Wait for all threads to complete
                for handle in handles {
                    handle.join().unwrap();
                }
            });
        },
    );

    // Stop the background threads
    running.store(false, Ordering::Relaxed);
    sync_running.store(false, Ordering::Relaxed);
    background_thread.join().unwrap();
    sync_thread.join().unwrap();

    group.finish();
}

criterion_group!(benches, bench_wal_insertion);
criterion_main!(benches);
