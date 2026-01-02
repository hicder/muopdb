use std::cell::UnsafeCell;
use std::os::fd::RawFd;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use dashmap::DashMap;
use log::info;
use parking_lot::Mutex;

/// High-performance asynchronous I/O engine using Linux io_uring.
///
/// This engine provides efficient file reading by leveraging Linux's io_uring
/// interface, which enables asynchronous operations without traditional
/// thread-per-request overhead. It uses a leader-follower pattern where
/// the submitting thread also handles completion queue processing.
///
/// The engine is designed for high-throughput scenarios where many concurrent
/// file reads are needed, such as database systems reading from disk.
///
/// # Architecture - Leader-Follower Pattern
///
/// ```text
/// ┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
/// │  Thread 1       │────▶│  submit_read     │────▶│  io_uring       │
/// │  (Leader)       │     │  (inline submit) │     │  (kernel)       │
/// └─────────────────┘     └──────────────────┘     └─────────────────┘
///                                                         │
///                                                         ▼
///                         ┌──────────────────┐     ┌─────────────────┐
///                         │  In-Flight Map   │◀────│  Leader polls   │
///                         │  (DashMap)       │     │  & sends        │
///                         └──────────────────┘     └─────────────────┘
///                              │  ▲
///                              │  │ oneshot
///                              ▼  │
///                         ┌──────────────────┐
///                         │  Thread 2+       │
///                         │  (Followers)     │
///                         │  wait on Notify  │
///                         └──────────────────┘
/// ```
///
/// # Leader-Follower Pattern
///
/// - Each thread submitting an I/O request tries to become the **leader**
/// - The leader acquires the `leader_lock`, submits to io_uring, and processes
///   all available completions
/// - Followers wait via `tokio::select!` on either their response channel
///   or a notification from the leader
/// - This eliminates the 1ms sleep latency from the background thread design
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
    uring: Arc<InnerUring>,
    inflight: Arc<DashMap<u64, InFlightEntry>>,
    completion_notify: Arc<tokio::sync::Notify>,
}

struct InnerUring {
    ring: UnsafeCell<io_uring::IoUring>,
    sq_lock: Mutex<()>,
    cq_lock: Mutex<()>,
}

// Safety: We protect access to the ring via SQ and CQ locks.
// io_uring queues are designed to be accessed independently.
unsafe impl Send for InnerUring {}
unsafe impl Sync for InnerUring {}

impl InnerUring {
    fn new(depth: u32) -> Result<Self> {
        let ring = io_uring::IoUring::new(depth)
            .map_err(|e| anyhow!("Failed to create io_uring: {}", e))?;
        Ok(Self {
            ring: UnsafeCell::new(ring),
            sq_lock: Mutex::new(()),
            cq_lock: Mutex::new(()),
        })
    }

    fn push_and_submit(&self, sqe: &io_uring::squeue::Entry) -> Result<()> {
        let _guard = self.sq_lock.lock();
        unsafe {
            let ring = &mut *self.ring.get();
            ring.submission()
                .push(sqe)
                .map_err(|_| anyhow!("io_uring submission queue is full"))?;
            ring.submit()
                .map_err(|e| anyhow!("io_uring submit failed: {}", e))?;
        }
        Ok(())
    }

    fn try_process_completions<F>(&self, mut f: F, target_id: Option<u64>) -> (bool, bool)
    where
        F: FnMut(io_uring::cqueue::Entry),
    {
        let Some(_guard) = self.cq_lock.try_lock() else {
            return (false, false);
        };
        let mut found = false;
        unsafe {
            let ring = &mut *self.ring.get();
            let mut cq = ring.completion();
            while let Some(cqe) = cq.next() {
                let id = cqe.user_data();
                f(cqe);
                if let Some(target) = target_id {
                    if id == target {
                        found = true;
                        break;
                    }
                }
            }
        }
        (true, found)
    }
}

// AtomicRefCell is Send/Sync if the inner type is Send/Sync.
// io_uring::IoUring is Send/Sync.

/// A cloneable handle for submitting read requests to the UringEngine.
///
/// This handle can be safely cloned and shared across multiple threads/tasks,
/// allowing concurrent read submissions without needing to reference the
/// original engine. The handle is lightweight and contains only the references
/// needed for submitting requests.
///
/// The handle maintains:
/// - References to the io_uring instance
/// - An atomic counter for generating unique request IDs
/// - The leader lock and completion notify for the leader-follower pattern
///
/// # Thread Safety
///
/// This struct is safe to use from multiple async tasks concurrently.
/// The internal atomic counter ensures request IDs are unique even under
/// concurrent access.
#[derive(Clone)]
pub struct UringEngineHandle {
    next_id: Arc<AtomicU64>,
    uring: Arc<InnerUring>,
    inflight: Arc<DashMap<u64, InFlightEntry>>,
    completion_notify: Arc<tokio::sync::Notify>,
}

enum InFlightEntry {
    Read {
        response_tx: tokio::sync::oneshot::Sender<Result<Vec<u8>>>,
        buffer: Pin<Box<Vec<u8>>>,
    },
    #[allow(dead_code)]
    Write {
        response_tx: tokio::sync::oneshot::Sender<Result<()>>,
        buffer: Pin<Box<Vec<u8>>>,
    },
}

/// Process a completion queue entry and send the result to the waiting task.
fn process_cqe(cqe: io_uring::cqueue::Entry, inflight: &DashMap<u64, InFlightEntry>) {
    let id = cqe.user_data();

    let entry = inflight.remove(&id);
    if entry.is_none() {
        log::warn!("Received completion for unknown request id: {}", id);
        return;
    }

    let (_key, entry) = entry.unwrap();

    match entry {
        InFlightEntry::Read {
            response_tx,
            buffer,
        } => {
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
        InFlightEntry::Write {
            response_tx,
            buffer: _,
        } => {
            let result = if cqe.result() < 0 {
                Err(anyhow!(
                    "uring write failed for request {}: {}",
                    id,
                    cqe.result()
                ))
            } else {
                Ok(())
            };

            let _ = response_tx.send(result);
        }
    }
}

impl UringEngine {
    /// Creates a new UringEngine with the specified queue depth.
    ///
    /// The queue depth determines the maximum number of pending submissions
    /// the io_uring instance can handle. A higher value allows more
    /// concurrent operations but consumes more kernel memory.
    ///
    /// The engine uses a leader-follower pattern where the thread submitting
    /// I/O requests also handles completion processing, eliminating the need
    /// for a dedicated background thread.
    ///
    /// # Arguments
    ///
    /// * `queue_depth` - Maximum number of entries in the io_uring submission queue
    ///
    /// # Panics
    ///
    /// Panics if creating the io_uring instance fails.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Create an engine with 512 entry queue depth
    /// let engine = UringEngine::new(512);
    /// ```
    pub fn new(queue_depth: u32) -> Self {
        info!("Creating UringEngine with queue depth {}", queue_depth);
        let uring = InnerUring::new(queue_depth).unwrap();

        let inflight: DashMap<u64, InFlightEntry> = DashMap::new();
        let inflight = Arc::new(inflight);
        let next_id = Arc::new(AtomicU64::new(0));
        let completion_notify = Arc::new(tokio::sync::Notify::new());

        UringEngine {
            next_id,
            uring: Arc::new(uring),
            inflight,
            completion_notify,
        }
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
            completion_notify: self.completion_notify.clone(),
        }
    }
}

impl UringEngineHandle {
    /// Submits an asynchronous read request to the io_uring engine.
    ///
    /// This function reads `length` bytes from the file descriptor starting
    /// at `offset` into the provided buffer. The operation is performed
    /// asynchronously using io_uring with a leader-follower pattern.
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

        let mut pinned_buffer = Box::pin(buffer);
        let sqe = io_uring::opcode::Read::new(
            io_uring::types::Fd(fd),
            pinned_buffer.as_mut().get_mut().as_mut_ptr(),
            length as u32,
        )
        .offset(offset)
        .build()
        .user_data(id);

        // Add to inflight map first
        self.inflight.insert(
            id,
            InFlightEntry::Read {
                response_tx,
                buffer: pinned_buffer,
            },
        );

        // Submit immediately (separate lock from completions)
        self.uring.push_and_submit(&sqe)?;

        // Wait for results
        self.wait_for_completion(id, response_rx, offset, length)
            .await
    }

    /// Leader-follower pattern: Try to become the leader or wait for notification.
    async fn wait_for_completion(
        &self,
        my_id: u64,
        mut response_rx: tokio::sync::oneshot::Receiver<Result<Vec<u8>>>,
        offset: u64,
        length: usize,
    ) -> Result<Vec<u8>> {
        loop {
            // Try to become the leader and process completions
            let (is_leader, found) = self.uring.try_process_completions(
                |cqe| {
                    process_cqe(cqe, &self.inflight);
                },
                Some(my_id),
            );

            if is_leader {
                // Notify all waiters (required after break in leader loop)
                self.completion_notify.notify_waiters();

                if found {
                    return response_rx
                        .await
                        .map_err(|e| anyhow!("Failed to receive read result: {}", e))?
                        .with_context(|| {
                            format!("uring read failed at offset {} (len {})", offset, length)
                        });
                }
            }

            // Not the leader: wait for either our response or a notification
            // If we WERE the leader, we shouldn't wait for notification (as we just sent one),
            // instead we yield and loop back to try to be leader again.
            tokio::select! {
                biased;
                result = &mut response_rx => {
                    return result
                        .map_err(|e| anyhow!("Failed to receive read result: {}", e))?
                        .with_context(|| {
                            format!("uring read failed at offset {} (len {})", offset, length)
                        });
                }
                _ = async {
                    if is_leader {
                        tokio::task::yield_now().await;
                    } else {
                        self.completion_notify.notified().await;
                    }
                } => {
                    // loop back
                }
            }
        }
    }

    pub async fn submit_write(&self, fd: RawFd, offset: u64, data: Vec<u8>) -> Result<()> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let (response_tx, response_rx) = tokio::sync::oneshot::channel();

        let mut pinned_buffer = Box::pin(data);
        let length = pinned_buffer.len();

        let sqe = io_uring::opcode::Write::new(
            io_uring::types::Fd(fd),
            pinned_buffer.as_mut().get_mut().as_ptr(),
            length as u32,
        )
        .offset(offset)
        .build()
        .user_data(id);

        // Add to inflight map first
        self.inflight.insert(
            id,
            InFlightEntry::Write {
                response_tx,
                buffer: pinned_buffer,
            },
        );

        // Submit immediately (separate lock from completions)
        self.uring.push_and_submit(&sqe)?;

        // Wait for results
        self.wait_for_completion_write(id, response_rx, offset, length)
            .await
    }

    /// Leader-follower pattern for write operations.
    async fn wait_for_completion_write(
        &self,
        my_id: u64,
        mut response_rx: tokio::sync::oneshot::Receiver<Result<()>>,
        offset: u64,
        length: usize,
    ) -> Result<()> {
        loop {
            // Try to become the leader and process completions
            let (is_leader, found) = self.uring.try_process_completions(
                |cqe| {
                    process_cqe(cqe, &self.inflight);
                },
                Some(my_id),
            );

            if is_leader {
                self.completion_notify.notify_waiters();

                if found {
                    return response_rx
                        .await
                        .map_err(|e| anyhow!("Failed to receive write result: {}", e))?
                        .with_context(|| {
                            format!("uring write failed at offset {} (len {})", offset, length)
                        });
                }
            }

            tokio::select! {
                biased;
                result = &mut response_rx => {
                    return result
                        .map_err(|e| anyhow!("Failed to receive write result: {}", e))?
                        .with_context(|| {
                            format!("uring write failed at offset {} (len {})", offset, length)
                        });
                }
                _ = async {
                    if is_leader {
                        tokio::task::yield_now().await;
                    } else {
                        self.completion_notify.notified().await;
                    }
                } => {
                    // loop back
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::{Read, Write};
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

    #[tokio::test]
    async fn test_write_basic() {
        let temp_dir = TempDir::new("uring_engine_test").unwrap();
        let test_file_path = temp_dir.path().join("test_write.txt");

        let engine = UringEngine::new(256);
        let handle = engine.handle();

        let file = std::fs::File::create(&test_file_path).unwrap();
        let fd = file.as_raw_fd();

        handle
            .submit_write(fd, 0, b"Hello, World!".to_vec())
            .await
            .unwrap();

        drop(file);

        let mut read_file = File::open(&test_file_path).unwrap();
        let mut buffer = Vec::new();
        read_file.read_to_end(&mut buffer).unwrap();
        assert_eq!(&buffer, b"Hello, World!");
    }

    #[tokio::test]
    async fn test_write_from_offset() {
        let temp_dir = TempDir::new("uring_engine_test").unwrap();
        let test_file_path = temp_dir.path().join("test_write_offset.txt");

        let engine = UringEngine::new(256);
        let handle = engine.handle();

        let file = std::fs::File::create(&test_file_path).unwrap();
        let fd = file.as_raw_fd();

        handle
            .submit_write(fd, 0, b"0000000000".to_vec())
            .await
            .unwrap();

        handle.submit_write(fd, 3, b"ABCDE".to_vec()).await.unwrap();

        drop(file);

        let mut read_file = File::open(&test_file_path).unwrap();
        let mut buffer = Vec::new();
        read_file.read_to_end(&mut buffer).unwrap();
        assert_eq!(&buffer, b"000ABCDE00");
    }

    #[tokio::test]
    async fn test_write_multiple() {
        let temp_dir = TempDir::new("uring_engine_test").unwrap();
        let test_file_path = temp_dir.path().join("test_write_multiple.txt");

        let engine = UringEngine::new(256);
        let handle = engine.handle();

        let file = std::fs::File::create(&test_file_path).unwrap();
        let fd = file.as_raw_fd();

        handle
            .submit_write(fd, 0, b"Hello, ".to_vec())
            .await
            .unwrap();

        handle
            .submit_write(fd, 7, b"World!".to_vec())
            .await
            .unwrap();

        drop(file);

        let mut read_file = File::open(&test_file_path).unwrap();
        let mut buffer = Vec::new();
        read_file.read_to_end(&mut buffer).unwrap();
        assert_eq!(&buffer, b"Hello, World!");
    }

    #[tokio::test]
    async fn test_write_large() {
        let temp_dir = TempDir::new("uring_engine_test").unwrap();
        let test_file_path = temp_dir.path().join("test_write_large.txt");

        let engine = UringEngine::new(256);
        let handle = engine.handle();

        let file = std::fs::File::create(&test_file_path).unwrap();
        let fd = file.as_raw_fd();

        let data = vec![42u8; 16384];
        handle.submit_write(fd, 0, data).await.unwrap();

        drop(file);

        let mut read_file = File::open(&test_file_path).unwrap();
        let mut buffer = Vec::new();
        read_file.read_to_end(&mut buffer).unwrap();
        assert_eq!(buffer.len(), 16384);
        assert!(buffer.iter().all(|&b| b == 42));
    }

    #[tokio::test]
    async fn test_write_empty() {
        let temp_dir = TempDir::new("uring_engine_test").unwrap();
        let test_file_path = temp_dir.path().join("test_write_empty.txt");

        let engine = UringEngine::new(256);
        let handle = engine.handle();

        let file = std::fs::File::create(&test_file_path).unwrap();
        let fd = file.as_raw_fd();

        handle.submit_write(fd, 0, vec![]).await.unwrap();

        drop(file);

        let mut read_file = File::open(&test_file_path).unwrap();
        let mut buffer = Vec::new();
        read_file.read_to_end(&mut buffer).unwrap();
        assert_eq!(buffer.len(), 0);
    }

    #[tokio::test]
    async fn test_concurrent_writes() {
        let temp_dir = TempDir::new("uring_engine_test").unwrap();
        let test_file_path = temp_dir.path().join("test_concurrent_writes.txt");

        let engine = UringEngine::new(256);
        let handle = engine.handle();

        let file = std::fs::File::create(&test_file_path).unwrap();
        let fd = file.as_raw_fd();

        let handle = std::sync::Arc::new(handle);
        let mut handles = Vec::new();
        for i in 0..2 {
            let handle = handle.clone();
            let data = format!("Write {}", i).into_bytes();
            let handle = tokio::spawn(async move {
                handle
                    .submit_write(fd, (i * 100) as u64, data)
                    .await
                    .unwrap()
            });
            handles.push(handle);
        }

        futures::future::join_all(handles).await;

        drop(file);

        let mut read_file = File::open(&test_file_path).unwrap();
        let mut buffer = Vec::new();
        read_file.read_to_end(&mut buffer).unwrap();
        assert_eq!(&buffer[0..7], b"Write 0");
        assert_eq!(&buffer[100..107], b"Write 1");
    }
}
