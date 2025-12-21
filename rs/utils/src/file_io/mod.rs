use std::fs::Metadata;

/// File I/O abstraction layer providing async file operations.
///
/// This module defines a trait-based abstraction for file operations,
/// allowing different storage backends to be used interchangeably.
use anyhow::Result;
use async_trait::async_trait;

pub mod standard_file;

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
}
