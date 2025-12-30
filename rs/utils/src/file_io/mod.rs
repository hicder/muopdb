use std::fs::Metadata;

/// File I/O abstraction layer providing async file operations.
///
/// This module defines a trait-based abstraction for file operations,
/// allowing different storage backends to be used interchangeably.
use anyhow::Result;
use async_trait::async_trait;

pub mod cached_file;
pub mod env;
pub mod mmap_file;
pub mod standard_file;

#[cfg(target_os = "linux")]
pub mod uring_engine;

#[cfg(target_os = "linux")]
pub mod uring_file;

pub use env::{Env, OpenAppendResult, OpenResult};
pub use standard_file::{AppendableStandardFile, StandardFile};
#[cfg(target_os = "linux")]
pub use uring_file::{AppendableUringFile, UringFile};

#[async_trait]
/// Trait for asynchronous file reading operations.
///
/// Provides a unified interface for reading file contents with support
/// for partial reads at specific offsets.
pub trait FileIO {
    /// Reads a contiguous byte range from the file.
    ///
    /// # Arguments
    /// * `offset` - Starting byte position in the file
    /// * `length` - Number of bytes to read
    ///
    /// # Returns
    /// A vector containing the requested byte range, or an error if the read fails.
    async fn read(&self, offset: u64, length: u64) -> Result<Vec<u8>>;

    /// Retrieves file metadata.
    ///
    /// # Returns
    /// Metadata information about the file (size, modification time, etc.).
    async fn metadata(&self) -> Result<Metadata>;

    /// Returns the length of the file in bytes.
    async fn file_length(&self) -> Result<u64>;

    /// Returns the block size used for I/O operations.
    ///
    /// For cached implementations, returns the configured block size.
    /// For others, returns a default of 4096 bytes.
    fn get_block_size(&self) -> usize {
        4096 // default implementation
    }
}

#[async_trait]
/// Trait for asynchronous file appending operations.
///
/// Provides a unified interface for appending data to files with support
/// for flushing and syncing to disk.
pub trait AppendableFileIO {
    /// Appends data to the end of the file.
    ///
    /// # Arguments
    /// * `data` - Byte slice to append
    ///
    /// # Returns
    /// Number of bytes written
    ///
    /// # Errors
    /// Returns an error if the write operation fails. If an error occurs,
    /// no data is written to the file.
    async fn append(&self, data: &[u8]) -> Result<u64>;

    /// Flushes buffered writes to disk.
    ///
    /// # Errors
    /// Returns an error if the flush operation fails.
    async fn flush(&self) -> Result<()>;

    /// Syncs file metadata and data to disk.
    ///
    /// # Errors
    /// Returns an error if the sync operation fails.
    async fn sync_all(&self) -> Result<()>;
}
