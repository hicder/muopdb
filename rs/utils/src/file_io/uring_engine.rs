use std::os::fd::RawFd;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use dashmap::DashMap;
use parking_lot::Mutex;

/// High-performance asynchronous I/O engine using Linux io_uring.
///
/// This engine provides efficient file reading by leveraging Linux's io_uring
/// interface, which enables asynchronous operations without traditional
/// thread-per-request overhead. It uses a single completion thread to poll
/// for finished operations while submissions happen synchronously from the
/// calling async task.
///
/// The engine is designed for high-throughput scenarios where many concurrent
/// file reads are needed, such as database systems reading from disk.
///
/// # Architecture
///
/// ```text
/// ┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
/// │  Tokio Task     │────▶│  submit_read     │────▶│  io_uring       │
/// │  (async read)   │     │  (inline submit) │     │  (kernel)       │
/// └─────────────────┘     └──────────────────┘     └─────────────────┘
///                                                         │
///                                                         ▼
///                         ┌──────────────────┐     ┌─────────────────┐
///                         │  In-Flight Map   │◀────│  Completion     │
///                         │  (DashMap)       │     │  Thread         │
///                         └──────────────────┘     └─────────────────┘
/// ```
///
/// # Example
///
/// ```ignore
/// use std::fs::File;
/// use std::os::fd::AsRawFd;
///
/// let engine = UringEngine::new(256);
/// let handle = engine.handle();
///
/// let file = File::open("data.bin")?;
/// let fd = file.as_raw_fd();
///
/// // Read asynchronously from the file
/// let data = handle.submit_read(fd, 0, 1024, vec![0; 1024]).await?;
/// ```
pub struct UringEngine {
    next_id: Arc<AtomicU64>,
    uring: Arc<Mutex<io_uring::IoUring>>,
    inflight: Arc<DashMap<u64, InFlightEntry>>,
}

/// A cloneable handle for submitting read requests to the UringEngine.
///
/// This handle can be safely cloned and shared across multiple threads/tasks,
/// allowing concurrent read submissions without needing to reference the
/// original engine. The handle is lightweight and contains only the channels
/// needed for submitting requests.
///
/// The handle maintains:
/// - A submission channel for sending read requests to the engine
/// - An atomic counter for generating unique request IDs
///
/// # Thread Safety
///
/// This struct is safe to use from multiple async tasks concurrently.
/// The internal atomic counter ensures request IDs are unique even under
/// concurrent access.
#[derive(Clone)]
pub struct UringEngineHandle {
    next_id: Arc<AtomicU64>,
    uring: Arc<Mutex<io_uring::IoUring>>,
    inflight: Arc<DashMap<u64, InFlightEntry>>,
}

struct InFlightEntry {
    response_tx: tokio::sync::oneshot::Sender<Result<Vec<u8>>>,
    buffer: Pin<Box<Vec<u8>>>,
}

impl UringEngine {
    /// Creates a new UringEngine with the specified queue depth.
    ///
    /// The queue depth determines the maximum number of pending submissions
    /// the io_uring instance can handle. A higher value allows more
    /// concurrent operations but consumes more kernel memory.
    ///
    /// This function spawns a single background completion thread that polls
    /// for finished I/O operations. Read submissions happen synchronously
    /// from the calling async task.
    ///
    /// # Arguments
    ///
    /// * `queue_depth` - Maximum number of entries in the io_uring submission queue
    ///
    /// # Panics
    ///
    /// Panics if creating the io_uring instance fails or if thread spawning fails.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Create an engine with 512 entry queue depth
    /// let engine = UringEngine::new(512);
    /// ```
    pub fn new(queue_depth: u32) -> Self {
        let uring = Arc::new(Mutex::new(
            io_uring::IoUring::new(queue_depth)
                .map_err(|e| anyhow!("Failed to create io_uring: {}", e))
                .unwrap(),
        ));

        let inflight: DashMap<u64, InFlightEntry> = DashMap::new();
        let inflight = Arc::new(inflight);
        let next_id = Arc::new(AtomicU64::new(0));

        let uring_clone = uring.clone();
        let inflight_clone = inflight.clone();

        thread::spawn(move || {
            Self::completion_thread_loop(uring_clone, inflight_clone);
        });

        UringEngine {
            next_id,
            uring,
            inflight,
        }
    }

    fn completion_thread_loop(
        uring: Arc<Mutex<io_uring::IoUring>>,
        inflight: Arc<DashMap<u64, InFlightEntry>>,
    ) {
        loop {
            let mut uring_guard = uring.lock();
            uring_guard.submit().ok();
            let mut has_completions = false;
            while let Some(cqe) = uring_guard.completion().next() {
                Self::process_cqe(cqe, &inflight);
                has_completions = true;
            }

            if !has_completions {
                drop(uring_guard);
                std::thread::sleep(Duration::from_millis(1));
            }
        }
    }

    fn process_cqe(cqe: io_uring::cqueue::Entry, inflight: &DashMap<u64, InFlightEntry>) {
        let id = cqe.user_data();

        let entry = inflight.remove(&id);
        if entry.is_none() {
            log::warn!("Received completion for unknown request id: {}", id);
            return;
        }

        let (
            _key,
            InFlightEntry {
                response_tx,
                buffer,
            },
        ) = entry.unwrap();

        let result = if cqe.result() < 0 {
            Err(anyhow!(
                "uring read failed for request {}: {}",
                id,
                cqe.result()
            ))
        } else {
            let bytes_read = cqe.result() as usize;
            let boxed_buffer = Pin::into_inner(buffer);
            let mut buffer = *boxed_buffer;
            buffer.truncate(bytes_read);
            Ok(buffer)
        };

        let _ = response_tx.send(result);
    }

    /// Creates a handle for submitting read requests to this engine.
    ///
    /// The returned handle can be safely cloned and shared across threads.
    /// Each handle operates independently but submits to the same underlying
    /// io_uring engine, enabling distributed read submissions.
    ///
    /// # Returns
    ///
    /// A new `UringEngineHandle` linked to this engine.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let engine = UringEngine::new(256);
    /// let handle = engine.handle();
    ///
    /// // Clone the handle for use in different tasks
    /// let handle_clone = handle.clone();
    /// ```
    pub fn handle(&self) -> UringEngineHandle {
        UringEngineHandle {
            next_id: self.next_id.clone(),
            uring: self.uring.clone(),
            inflight: self.inflight.clone(),
        }
    }
}

impl UringEngineHandle {
    /// Submits an asynchronous read request to the io_uring engine.
    ///
    /// This function reads `length` bytes from the file descriptor starting
    /// at `offset` into the provided buffer. The operation is performed
    /// asynchronously using io_uring, and the result is returned as a future.
    ///
    /// # Arguments
    ///
    /// * `fd` - Raw file descriptor to read from (obtained via `AsRawFd`)
    /// * `offset` - Byte offset in the file to start reading from
    /// * `length` - Number of bytes to read
    /// * `buffer` - Pre-allocated buffer to read into (must be at least `length` bytes)
    ///
    /// # Returns
    ///
    /// A future that resolves to the read data, trimmed to the actual number
    /// of bytes read. If fewer bytes are read than requested, the buffer
    /// will be truncated.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The request cannot be sent to the engine (channel closed)
    /// - The read operation fails (I/O error from kernel)
    /// - The oneshot channel is dropped prematurely
    ///
    /// # Example
    ///
    /// ```ignore
    /// use std::fs::File;
    /// use std::os::fd::AsRawFd;
    ///
    /// let file = File::open("data.bin")?;
    /// let fd = file.as_raw_fd();
    ///
    /// // Read first 1KB of the file
    /// let mut buffer = vec![0u8; 1024];
    /// let result = handle.submit_read(fd, 0, 1024, buffer).await?;
    /// println!("Read {} bytes", result.len());
    /// ```
    pub async fn submit_read(
        &self,
        fd: RawFd,
        offset: u64,
        length: usize,
        buffer: Vec<u8>,
    ) -> Result<Vec<u8>> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let (response_tx, response_rx) = tokio::sync::oneshot::channel();

        // Pin the buffer to ensure it doesn't move
        let mut pinned_buffer = Box::pin(buffer);
        let sqe = io_uring::opcode::Read::new(
            io_uring::types::Fd(fd),
            pinned_buffer.as_mut().get_mut().as_mut_ptr(),
            length as u32,
        )
        .offset(offset)
        .build()
        .user_data(id);

        {
            let mut uring_guard = self.uring.lock();
            unsafe {
                uring_guard.submission().push(&sqe).unwrap();
            }

            self.inflight.insert(
                id,
                InFlightEntry {
                    response_tx,
                    buffer: pinned_buffer,
                },
            );

            if let Err(e) = uring_guard.submit() {
                log::error!("io_uring submit error: {}", e);
            }
        }
        response_rx
            .await
            .map_err(|e| anyhow!("Failed to receive read result: {}", e))?
            .with_context(|| format!("uring read failed at offset {} (len {})", offset, length))
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::Write;
    use std::os::fd::AsRawFd;

    use tempdir::TempDir;

    use super::*;

    #[test]
    fn test_new_and_drop() {
        let _engine = UringEngine::new(256);
    }

    #[test]
    fn test_handle_clone() {
        let engine = UringEngine::new(256);
        let handle1 = engine.handle();
        let handle2 = handle1.clone();
        drop(engine);
        drop(handle1);
        drop(handle2);
    }

    #[tokio::test]
    async fn test_read_basic() {
        let temp_dir = TempDir::new("uring_engine_test").unwrap();
        let test_file_path = temp_dir.path().join("test.txt");
        let test_content = b"Hello, World!";
        let mut file = File::create(&test_file_path).unwrap();
        file.write_all(test_content).unwrap();

        let engine = UringEngine::new(256);
        let handle = engine.handle();

        let file = std::fs::File::open(&test_file_path).unwrap();
        let fd = file.as_raw_fd();

        let result = handle.submit_read(fd, 0, 5, vec![0; 5]).await.unwrap();
        assert_eq!(&result, b"Hello");

        drop(file);
    }

    #[tokio::test]
    async fn test_read_from_offset() {
        let temp_dir = TempDir::new("uring_engine_test").unwrap();
        let test_file_path = temp_dir.path().join("test.txt");
        let test_content = b"ABCDEFGHIJ";
        let mut file = File::create(&test_file_path).unwrap();
        file.write_all(test_content).unwrap();

        let engine = UringEngine::new(256);
        let handle = engine.handle();

        let file = std::fs::File::open(&test_file_path).unwrap();
        let fd = file.as_raw_fd();

        let result = handle.submit_read(fd, 3, 4, vec![0; 4]).await.unwrap();
        assert_eq!(&result, b"DEFG");

        drop(file);
    }

    #[tokio::test]
    async fn test_read_entire_file() {
        let temp_dir = TempDir::new("uring_engine_test").unwrap();
        let test_file_path = temp_dir.path().join("test.txt");
        let test_content = b"Full content here";
        let mut file = File::create(&test_file_path).unwrap();
        file.write_all(test_content).unwrap();

        let engine = UringEngine::new(256);
        let handle = engine.handle();

        let file = std::fs::File::open(&test_file_path).unwrap();
        let fd = file.as_raw_fd();

        let result = handle
            .submit_read(fd, 0, test_content.len(), vec![0; test_content.len()])
            .await
            .unwrap();
        assert_eq!(&result, test_content);

        drop(file);
    }

    #[tokio::test]
    async fn test_concurrent_reads() {
        let temp_dir = TempDir::new("uring_engine_test").unwrap();
        let test_file_path = temp_dir.path().join("test.txt");
        let test_content = b"ABCDEFGHIJ";
        let mut file = File::create(&test_file_path).unwrap();
        file.write_all(test_content).unwrap();

        let engine = UringEngine::new(256);
        let handle = engine.handle();

        let file = std::fs::File::open(&test_file_path).unwrap();
        let fd = file.as_raw_fd();

        let handle = std::sync::Arc::new(handle);
        let mut handles = Vec::new();
        for i in 0..2 {
            let fd = fd;
            let handle = handle.clone();
            let handle = tokio::spawn(async move {
                handle
                    .submit_read(fd, (i * 2) as u64, 2, vec![0; 2])
                    .await
                    .unwrap()
            });
            handles.push(handle);
        }

        let results: Vec<Vec<u8>> = futures::future::join_all(handles)
            .await
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert_eq!(results[0], b"AB");
        assert_eq!(results[1], b"CD");

        drop(file);
    }

    #[tokio::test]
    async fn test_read_large_file() {
        let temp_dir = TempDir::new("uring_engine_test").unwrap();
        let test_file_path = temp_dir.path().join("large.txt");
        let test_content = vec![42u8; 16384];
        let mut file = File::create(&test_file_path).unwrap();
        file.write_all(&test_content).unwrap();

        let engine = UringEngine::new(256);
        let handle = engine.handle();

        let file = std::fs::File::open(&test_file_path).unwrap();
        let fd = file.as_raw_fd();

        let result = handle
            .submit_read(fd, 0, 16384, vec![0; 16384])
            .await
            .unwrap();
        assert_eq!(result.len(), 16384);
        assert!(result.iter().all(|&b| b == 42));

        drop(file);
    }

    #[tokio::test]
    async fn test_read_empty_file() {
        let temp_dir = TempDir::new("uring_engine_test").unwrap();
        let test_file_path = temp_dir.path().join("empty.txt");
        let mut file = File::create(&test_file_path).unwrap();
        file.write_all(b"").unwrap();

        let engine = UringEngine::new(256);
        let handle = engine.handle();

        let file = std::fs::File::open(&test_file_path).unwrap();
        let fd = file.as_raw_fd();

        let result = handle.submit_read(fd, 0, 0, vec![0; 0]).await.unwrap();
        assert_eq!(result.len(), 0);

        drop(file);
    }

    #[tokio::test]
    async fn test_shutdown_with_pending() {
        let temp_dir = TempDir::new("uring_engine_test").unwrap();
        let test_file_path = temp_dir.path().join("test.txt");
        let test_content = b"Test content for shutdown";
        let mut file = File::create(&test_file_path).unwrap();
        file.write_all(test_content).unwrap();

        let engine = UringEngine::new(256);
        let handle = engine.handle();

        let file = std::fs::File::open(&test_file_path).unwrap();
        let fd = file.as_raw_fd();

        let result = handle.submit_read(fd, 0, 5, vec![0; 5]).await.unwrap();
        assert_eq!(&result, b"Test ");

        drop(file);
        drop(engine);
        drop(handle);
    }
}
