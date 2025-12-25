use std::fs::Metadata;
use std::os::fd::AsRawFd;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use log::info;

use crate::file_io::uring_engine::UringEngineHandle;
use crate::file_io::FileIO;

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
            std::mem::take(&mut buffers[idx])
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

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::Write;

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
}
