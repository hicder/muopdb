use std::fs::Metadata;
use std::os::fd::AsRawFd;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use log::info;
use parking_lot::Mutex as ParkingLotMutex;

use crate::file_io::uring_engine::UringEngineHandle;
use crate::file_io::{AppendableFileIO, FileIO};

/// A file wrapper providing asynchronous I/O using io_uring.
///
/// This struct wraps a standard file handle and provides an async `read`
/// interface backed by the `UringEngine`. It implements the `FileIO` trait,
/// making it compatible with the rest of the file I/O abstraction layer.
///
/// # Buffer Management
///
/// `UringFile` maintains an internal pool of buffers to reduce allocations
/// for repeated reads. Buffers are reused across read operations when they
/// are large enough to accommodate new requests. This is particularly
/// beneficial when reading fixed-size chunks from a file.
///
/// # Thread Safety
///
/// This struct uses `Arc` for the underlying file and `Mutex` for buffer
/// management, making it safe to use from multiple async tasks concurrently.
/// Multiple readers can share the same `UringFile` instance.
///
/// # Example
///
/// ```ignore
/// use std::sync::Arc;
/// use crate::file_io::uring_engine::UringEngine;
///
/// let engine = UringEngine::new(256);
/// let handle = engine.handle();
///
/// // Open a file for async reading
/// let file = UringFile::new("data.bin", handle).await?;
///
/// // Read data asynchronously
/// let chunk = file.read(0, 1024).await?;
/// let metadata = file.metadata().await?;
/// ```
pub struct UringFile {
    file: Arc<std::fs::File>,
    file_length: u64,
    engine: UringEngineHandle,
    buffers: Mutex<Vec<Vec<u8>>>,
}

impl UringFile {
    /// Opens a file for asynchronous reading using io_uring.
    ///
    /// This function opens the file at the given path and prepares it for
    /// async reads through the provided engine handle. The file is opened
    /// read-only and its metadata is immediately fetched to determine
    /// the total file length.
    ///
    /// # Arguments
    ///
    /// * `path` - The file system path to open
    /// * `engine` - A handle to the `UringEngine` that will perform async reads
    ///
    /// # Returns
    ///
    /// A `Result` containing the new `UringFile` instance, or an error if
    /// the file cannot be opened or its metadata cannot be read.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file does not exist or cannot be opened
    /// - File metadata cannot be retrieved
    ///
    /// # Example
    ///
    /// ```ignore
    /// use crate::file_io::uring_engine::UringEngine;
    ///
    /// let engine = UringEngine::new(256);
    /// let uring_file = UringFile::new("/path/to/file.bin", engine.handle())
    ///     .await?;
    /// ```
    pub async fn new(path: &str, engine: UringEngineHandle) -> Result<Self> {
        info!("[URING] Opening file: {}", path);
        let file =
            std::fs::File::open(path).with_context(|| format!("Failed to open file: {}", path))?;

        let metadata = file
            .metadata()
            .with_context(|| format!("Failed to get metadata for: {}", path))?;
        let file_length = metadata.len();

        Ok(UringFile {
            file: Arc::new(file),
            file_length,
            engine,
            buffers: Mutex::new(Vec::new()),
        })
    }

    fn get_buffer(&self, length: usize) -> Vec<u8> {
        info!("[URING] Getting buffer of size {}", length);
        let mut buffers = self.buffers.lock().unwrap();
        if let Some(idx) = buffers.iter().position(|b| b.len() >= length) {
            buffers.swap_remove(idx)
        } else {
            vec![0; length]
        }
    }

    fn return_buffer(&self, buffer: Vec<u8>) {
        let mut buffers = self.buffers.lock().unwrap();
        buffers.push(buffer);
    }
}

#[async_trait::async_trait]
impl FileIO for UringFile {
    /// Reads a contiguous byte range from the file asynchronously.
    ///
    /// This method uses io_uring for efficient asynchronous I/O. If the
    /// length is zero, an empty vector is returned immediately without
    /// performing any I/O.
    ///
    /// The method reuses internal buffers when possible to reduce allocation
    /// overhead for repeated reads of similar sizes.
    ///
    /// # Arguments
    ///
    /// * `offset` - Zero-based byte offset to start reading from
    /// * `length` - Number of bytes to read (can be 0)
    ///
    /// # Returns
    ///
    /// A `Result` containing the read bytes, or an error if the read fails.
    /// The returned vector's length equals the actual bytes read, which may
    /// be less than requested if EOF is reached.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The io_uring operation fails
    /// - The file descriptor becomes invalid
    ///
    /// # Example
    ///
    /// ```ignore
    /// let file = UringFile::new("data.bin", handle).await?;
    ///
    /// // Read first 1KB
    /// let header = file.read(0, 1024).await?;
    ///
    /// // Read from offset 512
    /// let chunk = file.read(512, 256).await?;
    /// ```
    async fn read(&self, offset: u64, length: u64) -> Result<Vec<u8>> {
        info!("[URING] Reading {} bytes from offset {}", length, offset);
        if length == 0 {
            return Ok(vec![]);
        }

        let buffer = self.get_buffer(length as usize);
        let fd = self.file.as_raw_fd();

        let result = self
            .engine
            .submit_read(fd, offset, length as usize, buffer)
            .await;

        match result {
            Ok(buffer) => {
                self.return_buffer(buffer.clone());
                Ok(buffer)
            }
            Err(e) => {
                self.return_buffer(vec![0; length as usize]);
                Err(e)
            }
        }
    }

    /// Retrieves file metadata asynchronously.
    ///
    /// Returns metadata about the underlying file, including size,
    /// permissions, and timestamps.
    ///
    /// # Returns
    ///
    /// A `Result` containing the file metadata.
    ///
    /// # Errors
    ///
    /// Returns an error if metadata cannot be retrieved from the filesystem.
    async fn metadata(&self) -> Result<Metadata> {
        self.file.metadata().context("Failed to get file metadata")
    }

    /// Returns the total file size in bytes.
    ///
    /// This is a convenience method that returns the file length that was
    /// captured when the file was opened, avoiding an additional filesystem
    /// call.
    ///
    /// # Returns
    ///
    /// The file size as a 64-bit unsigned integer.
    ///
    /// # Note
    ///
    /// If the file is being modified by another process, this value may
    /// not reflect the current actual size. For accurate current size,
    /// use `metadata()` instead.
    async fn file_length(&self) -> Result<u64> {
        Ok(self.file_length)
    }
}

/// A file wrapper providing asynchronous I/O using io_uring for writing.
///
/// This struct wraps a standard file handle with write permissions and provides
/// an async `append` interface backed by the `UringEngine`. It implements the
/// `AppendableFileIO` trait, making it compatible with the file I/O
/// abstraction layer.
///
/// # Thread Safety
///
/// This struct uses `Arc` for the underlying file, `AtomicU64` for offset
/// tracking, and `Mutex` for write serialization, making it safe to use
/// from multiple async tasks concurrently.
pub struct AppendableUringFile {
    file: Arc<std::fs::File>,
    engine: UringEngineHandle,
    offset: AtomicU64,
    write_lock: ParkingLotMutex<()>,
}

impl AppendableUringFile {
    /// Opens a file for asynchronous writing using io_uring.
    ///
    /// This function opens the file at the given path and prepares it for
    /// async writes through the provided engine handle. The file must exist
    /// and have write permissions.
    ///
    /// # Arguments
    ///
    /// * `path` - The file system path to open
    /// * `engine` - A handle to the `UringEngine` that will perform async writes
    ///
    /// # Returns
    ///
    /// A `Result` containing the new `AppendableUringFile` instance, or an error
    /// if the file cannot be opened or its metadata cannot be read.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file does not exist or cannot be opened with write permissions
    /// - File metadata cannot be retrieved
    pub async fn new(path: &str, engine: UringEngineHandle) -> Result<Self> {
        info!("[URING] Opening file for writing: {}", path);
        let file = std::fs::OpenOptions::new()
            .write(true)
            .open(path)
            .with_context(|| format!("Failed to open file for writing: {}", path))?;

        let metadata = file
            .metadata()
            .with_context(|| format!("Failed to get metadata for: {}", path))?;
        let file_length = metadata.len();

        Ok(AppendableUringFile {
            file: Arc::new(file),
            engine,
            offset: AtomicU64::new(file_length),
            write_lock: ParkingLotMutex::new(()),
        })
    }
}

#[async_trait::async_trait]
impl AppendableFileIO for AppendableUringFile {
    /// Appends data to the end of the file asynchronously.
    ///
    /// This method uses io_uring for efficient asynchronous I/O. The write
    /// is performed at the current tracked offset, which is then updated.
    /// A mutex ensures concurrent writes are serialized.
    ///
    /// # Arguments
    ///
    /// * `data` - Byte slice to append
    ///
    /// # Returns
    ///
    /// A `Result` containing the number of bytes written, or an error if the
    /// write fails.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The io_uring operation fails
    /// - The file descriptor becomes invalid
    async fn append(&self, data: &[u8]) -> Result<u64> {
        let _lock = self.write_lock.lock();
        let offset = self.offset.fetch_add(data.len() as u64, Ordering::SeqCst);
        let fd = self.file.as_raw_fd();

        self.engine.submit_write(fd, offset, data.to_vec()).await?;

        Ok(data.len() as u64)
    }

    /// Flushes buffered writes to disk.
    ///
    /// # Errors
    ///
    /// Returns an error if the flush operation fails.
    async fn flush(&self) -> Result<()> {
        let file = self.file.clone();
        tokio::task::spawn_blocking(move || file.sync_data().context("Failed to sync data")).await?
    }

    /// Syncs file metadata and data to disk.
    ///
    /// # Errors
    ///
    /// Returns an error if the sync operation fails.
    async fn sync_all(&self) -> Result<()> {
        let file = self.file.clone();
        tokio::task::spawn_blocking(move || file.sync_all().context("Failed to sync all")).await?
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::{Read, Write};

    use tempdir::TempDir;

    use super::*;
    use crate::file_io::uring_engine::UringEngine;

    #[tokio::test]
    async fn test_read_basic() {
        let temp_dir = TempDir::new("uring_file_test").unwrap();
        let test_file_path = temp_dir.path().join("test.txt");
        let test_content = b"Hello, World!";
        let mut file = File::create(&test_file_path).unwrap();
        file.write_all(test_content).unwrap();

        let engine = UringEngine::new(256);
        let uring_file = UringFile::new(test_file_path.to_str().unwrap(), engine.handle())
            .await
            .unwrap();

        let result = uring_file.read(0, 5).await.unwrap();
        assert_eq!(&result, b"Hello");
    }

    #[tokio::test]
    async fn test_read_from_offset() {
        let temp_dir = TempDir::new("uring_file_test").unwrap();
        let test_file_path = temp_dir.path().join("test.txt");
        let test_content = b"ABCDEFGHIJ";
        let mut file = File::create(&test_file_path).unwrap();
        file.write_all(test_content).unwrap();

        let engine = UringEngine::new(256);
        let uring_file = UringFile::new(test_file_path.to_str().unwrap(), engine.handle())
            .await
            .unwrap();

        let result = uring_file.read(3, 4).await.unwrap();
        assert_eq!(&result, b"DEFG");
    }

    #[tokio::test]
    async fn test_read_entire_file() {
        let temp_dir = TempDir::new("uring_file_test").unwrap();
        let test_file_path = temp_dir.path().join("test.txt");
        let test_content = b"Full content here";
        let mut file = File::create(&test_file_path).unwrap();
        file.write_all(test_content).unwrap();

        let engine = UringEngine::new(256);
        let uring_file = UringFile::new(test_file_path.to_str().unwrap(), engine.handle())
            .await
            .unwrap();

        let result = uring_file.read(0, test_content.len() as u64).await.unwrap();
        assert_eq!(&result, test_content);
    }

    #[tokio::test]
    async fn test_read_last_bytes() {
        let temp_dir = TempDir::new("uring_file_test").unwrap();
        let test_file_path = temp_dir.path().join("test.txt");
        let test_content = b"0123456789";
        let mut file = File::create(&test_file_path).unwrap();
        file.write_all(test_content).unwrap();

        let engine = UringEngine::new(256);
        let uring_file = UringFile::new(test_file_path.to_str().unwrap(), engine.handle())
            .await
            .unwrap();

        let result = uring_file.read(7, 3).await.unwrap();
        assert_eq!(&result, b"789");
    }

    #[tokio::test]
    async fn test_file_length() {
        let temp_dir = TempDir::new("uring_file_test").unwrap();
        let test_file_path = temp_dir.path().join("test.txt");
        let test_content = vec![0u8; 1024];
        let mut file = File::create(&test_file_path).unwrap();
        file.write_all(&test_content).unwrap();

        let engine = UringEngine::new(256);
        let uring_file = UringFile::new(test_file_path.to_str().unwrap(), engine.handle())
            .await
            .unwrap();

        let length = uring_file.file_length().await.unwrap();
        assert_eq!(length, 1024);
    }

    #[tokio::test]
    async fn test_metadata() {
        let temp_dir = TempDir::new("uring_file_test").unwrap();
        let test_file_path = temp_dir.path().join("test.txt");
        let test_content = b"test data";
        let mut file = File::create(&test_file_path).unwrap();
        file.write_all(test_content).unwrap();

        let engine = UringEngine::new(256);
        let uring_file = UringFile::new(test_file_path.to_str().unwrap(), engine.handle())
            .await
            .unwrap();

        let metadata = uring_file.metadata().await.unwrap();
        assert_eq!(metadata.len(), test_content.len() as u64);
    }

    #[tokio::test]
    async fn test_concurrent_reads() {
        let temp_dir = TempDir::new("uring_file_test").unwrap();
        let test_file_path = temp_dir.path().join("test.txt");
        let test_content = b"ABCDEFGHIJ";
        let mut file = File::create(&test_file_path).unwrap();
        file.write_all(test_content).unwrap();

        let engine = UringEngine::new(256);
        let uring_file = UringFile::new(test_file_path.to_str().unwrap(), engine.handle())
            .await
            .unwrap();

        let uring_file = std::sync::Arc::new(uring_file);
        let mut handles = Vec::new();
        for i in 0..2 {
            let file = uring_file.clone();
            let handle = tokio::spawn(async move { file.read(i * 2, 2).await.unwrap() });
            handles.push(handle);
        }

        let results: Vec<Vec<u8>> = futures::future::join_all(handles)
            .await
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert_eq!(results[0], b"AB");
        assert_eq!(results[1], b"CD");
    }

    #[tokio::test]
    async fn test_read_large_file() {
        let temp_dir = TempDir::new("uring_file_test").unwrap();
        let test_file_path = temp_dir.path().join("large.txt");
        let test_content = vec![42u8; 16384];
        let mut file = File::create(&test_file_path).unwrap();
        file.write_all(&test_content).unwrap();

        let engine = UringEngine::new(256);
        let uring_file = UringFile::new(test_file_path.to_str().unwrap(), engine.handle())
            .await
            .unwrap();

        let result = uring_file.read(0, 16384).await.unwrap();
        assert_eq!(result.len(), 16384);
        assert!(result.iter().all(|&b| b == 42));
    }

    #[tokio::test]
    async fn test_read_empty_file() {
        let temp_dir = TempDir::new("uring_file_test").unwrap();
        let test_file_path = temp_dir.path().join("empty.txt");
        let mut file = File::create(&test_file_path).unwrap();
        file.write_all(b"").unwrap();

        let engine = UringEngine::new(256);
        let uring_file = UringFile::new(test_file_path.to_str().unwrap(), engine.handle())
            .await
            .unwrap();

        let length = uring_file.file_length().await.unwrap();
        assert_eq!(length, 0);
    }

    #[tokio::test]
    async fn test_appendable_append_basic() {
        let temp_dir = TempDir::new("uring_file_test").unwrap();
        let test_file_path = temp_dir.path().join("test_append.txt");

        let mut file = File::create(&test_file_path).unwrap();
        file.write_all(b"").unwrap();

        let engine = UringEngine::new(256);
        let appendable =
            AppendableUringFile::new(test_file_path.to_str().unwrap(), engine.handle())
                .await
                .unwrap();

        appendable.append(b"Hello, World!").await.unwrap();
        appendable.flush().await.unwrap();

        let mut read_file = File::open(&test_file_path).unwrap();
        let mut buffer = Vec::new();
        read_file.read_to_end(&mut buffer).unwrap();
        assert_eq!(&buffer, b"Hello, World!");
    }

    #[tokio::test]
    async fn test_appendable_append_multiple() {
        let temp_dir = TempDir::new("uring_file_test").unwrap();
        let test_file_path = temp_dir.path().join("test_append_multiple.txt");

        let mut file = File::create(&test_file_path).unwrap();
        file.write_all(b"").unwrap();

        let engine = UringEngine::new(256);
        let appendable =
            AppendableUringFile::new(test_file_path.to_str().unwrap(), engine.handle())
                .await
                .unwrap();

        appendable.append(b"Hello, ").await.unwrap();
        appendable.append(b"World!").await.unwrap();
        appendable.flush().await.unwrap();

        let mut read_file = File::open(&test_file_path).unwrap();
        let mut buffer = Vec::new();
        read_file.read_to_end(&mut buffer).unwrap();
        assert_eq!(&buffer, b"Hello, World!");
    }

    #[tokio::test]
    async fn test_appendable_append_empty() {
        let temp_dir = TempDir::new("uring_file_test").unwrap();
        let test_file_path = temp_dir.path().join("test_append_empty.txt");

        let mut file = File::create(&test_file_path).unwrap();
        file.write_all(b"").unwrap();

        let engine = UringEngine::new(256);
        let appendable =
            AppendableUringFile::new(test_file_path.to_str().unwrap(), engine.handle())
                .await
                .unwrap();

        let bytes_written = appendable.append(b"").await.unwrap();
        assert_eq!(bytes_written, 0);
        appendable.flush().await.unwrap();

        let metadata = std::fs::metadata(&test_file_path).unwrap();
        assert_eq!(metadata.len(), 0);
    }

    #[tokio::test]
    async fn test_appendable_append_large() {
        let temp_dir = TempDir::new("uring_file_test").unwrap();
        let test_file_path = temp_dir.path().join("test_append_large.txt");

        let mut file = File::create(&test_file_path).unwrap();
        file.write_all(b"").unwrap();

        let engine = UringEngine::new(256);
        let appendable =
            AppendableUringFile::new(test_file_path.to_str().unwrap(), engine.handle())
                .await
                .unwrap();

        let large_data = vec![42u8; 1024 * 1024];
        appendable.append(&large_data).await.unwrap();
        appendable.flush().await.unwrap();

        let metadata = std::fs::metadata(&test_file_path).unwrap();
        assert_eq!(metadata.len(), 1024 * 1024);
    }

    #[tokio::test]
    async fn test_appendable_flush() {
        let temp_dir = TempDir::new("uring_file_test").unwrap();
        let test_file_path = temp_dir.path().join("test_flush.txt");

        let mut file = File::create(&test_file_path).unwrap();
        file.write_all(b"").unwrap();

        let engine = UringEngine::new(256);
        let appendable =
            AppendableUringFile::new(test_file_path.to_str().unwrap(), engine.handle())
                .await
                .unwrap();

        appendable.append(b"Test data").await.unwrap();
        appendable.flush().await.unwrap();

        let mut read_file = File::open(&test_file_path).unwrap();
        let mut buffer = Vec::new();
        read_file.read_to_end(&mut buffer).unwrap();
        assert_eq!(&buffer, b"Test data");
    }

    #[tokio::test]
    async fn test_appendable_sync_all() {
        let temp_dir = TempDir::new("uring_file_test").unwrap();
        let test_file_path = temp_dir.path().join("test_sync_all.txt");

        let mut file = File::create(&test_file_path).unwrap();
        file.write_all(b"").unwrap();

        let engine = UringEngine::new(256);
        let appendable =
            AppendableUringFile::new(test_file_path.to_str().unwrap(), engine.handle())
                .await
                .unwrap();

        appendable.append(b"Sync test").await.unwrap();
        appendable.sync_all().await.unwrap();

        let mut read_file = File::open(&test_file_path).unwrap();
        let mut buffer = Vec::new();
        read_file.read_to_end(&mut buffer).unwrap();
        assert_eq!(&buffer, b"Sync test");
    }
}
