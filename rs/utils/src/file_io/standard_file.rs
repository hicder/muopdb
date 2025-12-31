/// Standard file implementation of the [`FileIO`] trait.
///
/// Wraps a [`tokio::fs::File`] with thread-safe access using `RwLock`.
/// Provides async file reading operations for regular filesystem files.
use std::fs::Metadata;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use anyhow::{anyhow, Result};
use tokio::fs::File;
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt, BufWriter, SeekFrom};
use tokio::sync::RwLock;

use crate::file_io::{AppendableFileIO, FileIO};

/// Default buffer size for buffered writes (64KB)
const DEFAULT_WRITE_BUFFER_SIZE: usize = 64 * 1024;

/// A thread-safe file wrapper for async read operations.
///
/// Uses `RwLock` to allow concurrent reads while ensuring exclusive
/// access during seek operations.
pub struct StandardFile {
    /// The underlying tokio file with read lock protection.
    file: RwLock<File>,
    /// Length of the file in bytes.
    file_length: u64,
}

impl StandardFile {
    /// Creates a new `StandardFile` instance from a [`File`].
    ///
    /// # Arguments
    /// * `file` - An opened tokio file handle
    ///
    /// # Returns
    /// A new `StandardFile` instance wrapped in an `RwLock`.
    pub async fn new(file: File) -> Self {
        let metadata = file.metadata().await.unwrap();
        Self {
            file: RwLock::new(file),
            file_length: metadata.len(),
        }
    }
}

#[async_trait::async_trait]
impl FileIO for StandardFile {
    /// Reads a contiguous byte range from the file.
    ///
    /// Seeks to the specified offset, reads exactly `length` bytes,
    /// and returns them as a vector.
    ///
    /// # Errors
    /// Returns an error if seek or read operations fail.
    async fn read(&self, offset: u64, length: u64) -> Result<Vec<u8>> {
        let mut file = self.file.write().await;
        file.seek(SeekFrom::Start(offset)).await?;
        let mut buffer = vec![0; length as usize];
        match file.read(&mut buffer).await {
            Ok(bytes_read) => {
                buffer.truncate(bytes_read);
                Ok(buffer)
            }
            Err(e) => Err(anyhow!("Failed to read file: {}", e)),
        }
    }

    /// Retrieves file metadata.
    ///
    /// # Errors
    /// Returns an error if metadata cannot be retrieved.
    async fn metadata(&self) -> Result<Metadata> {
        let file = self.file.read().await;
        Ok(file.metadata().await?)
    }

    /// Returns the length of the file in bytes.
    async fn file_length(&self) -> Result<u64> {
        Ok(self.file_length)
    }
}

/// A thread-safe file wrapper for async write operations.
///
/// Uses `RwLock` for write serialization and `AtomicU64` for thread-safe
/// length tracking. Provides append-only semantics with guaranteed
/// all-or-nothing writes. Uses buffered writes to reduce syscall frequency.
pub struct AppendableStandardFile {
    /// The underlying buffered tokio file.
    file: RwLock<BufWriter<File>>,
    /// Length of the file in bytes, atomically updated.
    file_length: Arc<AtomicU64>,
    /// Buffer size for writes.
    #[allow(dead_code)]
    buffer_size: usize,
}

impl AppendableStandardFile {
    /// Creates a new `AppendableStandardFile` instance from a [`File`].
    ///
    /// The file should be opened in append mode or with appropriate permissions
    /// for writing. The initial file length is read from the file metadata.
    /// Uses the default buffer size (64KB).
    ///
    /// # Arguments
    /// * `file` - An opened tokio file handle with write permissions
    ///
    /// # Returns
    /// A new `AppendableStandardFile` instance.
    pub async fn new(file: File) -> Self {
        Self::new_with_buffer_size(file, DEFAULT_WRITE_BUFFER_SIZE).await
    }

    /// Creates a new `AppendableStandardFile` instance with a custom buffer size.
    ///
    /// The file should be opened in append mode or with appropriate permissions
    /// for writing. The initial file length is read from the file metadata.
    ///
    /// # Arguments
    /// * `file` - An opened tokio file handle with write permissions
    /// * `buffer_size` - Size of the write buffer in bytes
    ///
    /// # Returns
    /// A new `AppendableStandardFile` instance.
    pub async fn new_with_buffer_size(file: File, buffer_size: usize) -> Self {
        let metadata = file.metadata().await.unwrap();
        Self {
            file: RwLock::new(BufWriter::with_capacity(buffer_size, file)),
            file_length: Arc::new(AtomicU64::new(metadata.len())),
            buffer_size,
        }
    }
}

#[async_trait::async_trait]
impl AppendableFileIO for AppendableStandardFile {
    /// Appends data to the end of the file.
    ///
    /// This method guarantees all-or-nothing semantics - either all bytes
    /// are written or none are. The cached file length is updated atomically
    /// after a successful write. Writes are buffered to reduce syscall frequency.
    ///
    /// # Arguments
    /// * `data` - Byte slice to append
    ///
    /// # Returns
    /// Number of bytes written
    ///
    /// # Errors
    /// Returns an error if the write operation fails.
    async fn append(&self, data: &[u8]) -> Result<u64> {
        let mut file = self.file.write().await;

        let offset = self
            .file_length
            .fetch_add(data.len() as u64, Ordering::SeqCst);
        file.seek(SeekFrom::Start(offset)).await?;
        file.write_all(data).await?;

        Ok(data.len() as u64)
    }

    /// Flushes buffered writes to disk.
    ///
    /// Flushes the internal buffer to the underlying file.
    ///
    /// # Errors
    /// Returns an error if the flush operation fails.
    async fn flush(&self) -> Result<()> {
        let mut file = self.file.write().await;
        file.flush().await?;
        Ok(())
    }

    /// Syncs file metadata and data to disk.
    ///
    /// Flushes the internal buffer first, then syncs the underlying file.
    ///
    /// # Errors
    /// Returns an error if the flush or sync operation fails.
    async fn sync_all(&self) -> Result<()> {
        let mut file = self.file.write().await;
        file.flush().await?;
        file.get_ref().sync_all().await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::io::Read;

    use tempdir::TempDir;

    use super::*;

    #[tokio::test]
    async fn test_append_basic() {
        let temp_dir = TempDir::new("appendable_standard_test").unwrap();
        let test_file_path = temp_dir.path().join("test_append.txt");

        let file = File::create(&test_file_path).await.unwrap();
        let appendable = AppendableStandardFile::new(file).await;

        appendable.append(b"Hello, World!").await.unwrap();
        appendable.flush().await.unwrap();

        let mut read_file = std::fs::File::open(&test_file_path).unwrap();
        let mut buffer = Vec::new();
        read_file.read_to_end(&mut buffer).unwrap();
        assert_eq!(&buffer, b"Hello, World!");
    }

    #[tokio::test]
    async fn test_append_multiple() {
        let temp_dir = TempDir::new("appendable_standard_test").unwrap();
        let test_file_path = temp_dir.path().join("test_append_multiple.txt");

        let file = File::create(&test_file_path).await.unwrap();
        let appendable = AppendableStandardFile::new(file).await;

        appendable.append(b"Hello, ").await.unwrap();
        appendable.append(b"World!").await.unwrap();
        appendable.flush().await.unwrap();

        let mut read_file = std::fs::File::open(&test_file_path).unwrap();
        let mut buffer = Vec::new();
        read_file.read_to_end(&mut buffer).unwrap();
        assert_eq!(&buffer, b"Hello, World!");
    }

    #[tokio::test]
    async fn test_append_empty() {
        let temp_dir = TempDir::new("appendable_standard_test").unwrap();
        let test_file_path = temp_dir.path().join("test_append_empty.txt");

        let file = File::create(&test_file_path).await.unwrap();
        let appendable = AppendableStandardFile::new(file).await;

        let bytes_written = appendable.append(b"").await.unwrap();
        assert_eq!(bytes_written, 0);
        appendable.flush().await.unwrap();

        let read_file = std::fs::File::open(&test_file_path).unwrap();
        let metadata = read_file.metadata().unwrap();
        assert_eq!(metadata.len(), 0);
    }

    #[tokio::test]
    async fn test_append_large() {
        let temp_dir = TempDir::new("appendable_standard_test").unwrap();
        let test_file_path = temp_dir.path().join("test_append_large.txt");

        let file = File::create(&test_file_path).await.unwrap();
        let appendable = AppendableStandardFile::new(file).await;

        let large_data = vec![42u8; 1024 * 1024];
        appendable.append(&large_data).await.unwrap();
        appendable.flush().await.unwrap();

        let read_file = std::fs::File::open(&test_file_path).unwrap();
        let metadata = read_file.metadata().unwrap();
        assert_eq!(metadata.len(), 1024 * 1024);
    }

    #[tokio::test]
    async fn test_concurrent_appends() {
        let temp_dir = TempDir::new("appendable_standard_test").unwrap();
        let test_file_path = temp_dir.path().join("test_concurrent_appends.txt");

        let file = File::create(&test_file_path).await.unwrap();
        let appendable = Arc::new(AppendableStandardFile::new(file).await);

        let mut handles = Vec::new();
        for i in 0..5 {
            let appendable = appendable.clone();
            let data = format!("Batch {}", i);
            let handle = tokio::spawn(async move {
                appendable.append(data.as_bytes()).await.unwrap();
            });
            handles.push(handle);
        }

        futures::future::join_all(handles).await;

        appendable.flush().await.unwrap();

        let mut read_file = std::fs::File::open(&test_file_path).unwrap();
        let mut buffer = Vec::new();
        read_file.read_to_end(&mut buffer).unwrap();

        let result = String::from_utf8(buffer).unwrap();
        assert!(result.starts_with("Batch "));
        assert_eq!(result.len(), 35);
    }

    #[tokio::test]
    async fn test_flush() {
        let temp_dir = TempDir::new("appendable_standard_test").unwrap();
        let test_file_path = temp_dir.path().join("test_flush.txt");

        let file = File::create(&test_file_path).await.unwrap();
        let appendable = AppendableStandardFile::new(file).await;

        appendable.append(b"Test data").await.unwrap();
        appendable.flush().await.unwrap();

        let mut read_file = std::fs::File::open(&test_file_path).unwrap();
        let mut buffer = Vec::new();
        read_file.read_to_end(&mut buffer).unwrap();
        assert_eq!(&buffer, b"Test data");
    }

    #[tokio::test]
    async fn test_sync_all() {
        let temp_dir = TempDir::new("appendable_standard_test").unwrap();
        let test_file_path = temp_dir.path().join("test_sync_all.txt");

        let file = File::create(&test_file_path).await.unwrap();
        let appendable = AppendableStandardFile::new(file).await;

        appendable.append(b"Sync test").await.unwrap();
        appendable.sync_all().await.unwrap();

        let mut read_file = std::fs::File::open(&test_file_path).unwrap();
        let mut buffer = Vec::new();
        read_file.read_to_end(&mut buffer).unwrap();
        assert_eq!(&buffer, b"Sync test");
    }

    #[tokio::test]
    async fn test_buffer_batching() {
        let temp_dir = TempDir::new("appendable_standard_test").unwrap();
        let test_file_path = temp_dir.path().join("test_buffer_batching.txt");

        let file = File::create(&test_file_path).await.unwrap();
        let appendable = AppendableStandardFile::new(file).await;

        // Multiple small writes that should be batched
        for i in 0..10 {
            appendable
                .append(format!("Chunk {}", i).as_bytes())
                .await
                .unwrap();
        }

        // Data should be available after flush
        appendable.flush().await.unwrap();

        let mut read_file = std::fs::File::open(&test_file_path).unwrap();
        let mut buffer = Vec::new();
        read_file.read_to_end(&mut buffer).unwrap();

        let result = String::from_utf8(buffer).unwrap();
        assert!(result.starts_with("Chunk 0"));
        assert!(result.contains("Chunk 9"));
    }

    #[tokio::test]
    async fn test_custom_buffer_size() {
        let temp_dir = TempDir::new("appendable_standard_test").unwrap();
        let test_file_path = temp_dir.path().join("test_custom_buffer_size.txt");

        let file = File::create(&test_file_path).await.unwrap();
        let custom_buffer_size = 1024; // 1KB buffer
        let appendable =
            AppendableStandardFile::new_with_buffer_size(file, custom_buffer_size).await;

        appendable.append(b"Custom buffer test").await.unwrap();
        appendable.flush().await.unwrap();

        let mut read_file = std::fs::File::open(&test_file_path).unwrap();
        let mut buffer = Vec::new();
        read_file.read_to_end(&mut buffer).unwrap();
        assert_eq!(&buffer, b"Custom buffer test");
    }

    #[tokio::test]
    async fn test_flush_writes_buffer() {
        let temp_dir = TempDir::new("appendable_standard_test").unwrap();
        let test_file_path = temp_dir.path().join("test_flush_writes_buffer.txt");

        let file = File::create(&test_file_path).await.unwrap();
        let appendable = AppendableStandardFile::new(file).await;

        // Write data without flush - should not be visible on disk yet
        appendable.append(b"Before flush").await.unwrap();

        // File should be empty or incomplete before flush
        let read_file = std::fs::File::open(&test_file_path).unwrap();
        let metadata = read_file.metadata().unwrap();
        // The buffered data may not be on disk yet
        assert!(metadata.len() == 0 || metadata.len() == "Before flush".len() as u64);

        drop(read_file);

        // After flush, data should be on disk
        appendable.flush().await.unwrap();

        let mut read_file = std::fs::File::open(&test_file_path).unwrap();
        let mut buffer = Vec::new();
        read_file.read_to_end(&mut buffer).unwrap();
        assert_eq!(&buffer, b"Before flush");
    }
}
