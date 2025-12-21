/// Standard file implementation of the [`FileIO`] trait.
///
/// Wraps a [`tokio::fs::File`] with thread-safe access using `RwLock`.
/// Provides async file reading operations for regular filesystem files.
use std::fs::Metadata;

use anyhow::{anyhow, Result};
use tokio::fs::File;
use tokio::io::{AsyncReadExt, AsyncSeekExt, SeekFrom};
use tokio::sync::RwLock;

use crate::file_io::FileIO;

/// A thread-safe file wrapper for async read operations.
///
/// Uses `RwLock` to allow concurrent reads while ensuring exclusive
/// access during seek operations.
pub struct StandardFile {
    /// The underlying tokio file with read lock protection.
    file: RwLock<File>,
}

impl StandardFile {
    /// Creates a new `StandardFile` instance from a [`File`].
    ///
    /// # Arguments
    /// * `file` - An opened tokio file handle
    ///
    /// # Returns
    /// A new `StandardFile` instance wrapped in an `RwLock`.
    pub fn new(file: File) -> Self {
        Self {
            file: RwLock::new(file),
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
}
