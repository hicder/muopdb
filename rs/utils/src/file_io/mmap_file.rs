use std::fs::Metadata;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use memmap2::Mmap;

use crate::file_io::FileIO;

/// Memory-mapped file implementation of the [`FileIO`] trait.
///
/// Provides zero-copy read operations by mapping the file into the process's
/// address space. This implementation is particularly efficient for random
/// access reads and leverages the OS page cache for performance.
pub struct MMapFileIO {
    mmap: Mmap,
    file: Arc<std::fs::File>,
    file_length: u64,
}

impl MMapFileIO {
    /// Creates a new `MMapFileIO` instance by memory-mapping the file at the given path.
    ///
    /// # Arguments
    /// * `path` - The system path to the file to be mapped
    ///
    /// # Returns
    /// A `Result` containing the `MMapFileIO` instance or an error if the file
    /// cannot be opened or mapped.
    pub fn new(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        let metadata = file.metadata()?;
        let file_length = metadata.len();

        // Safety: We are mapping a file that we just opened.
        // The file is opened in read-only mode by default with std::fs::File::open.
        let mmap = unsafe { Mmap::map(&file)? };

        Ok(Self {
            mmap,
            file: Arc::new(file),
            file_length,
        })
    }
}

#[async_trait]
impl FileIO for MMapFileIO {
    /// Reads a contiguous byte range from the memory-mapped file.
    ///
    /// # Arguments
    /// * `offset` - Starting byte position in the file
    /// * `length` - Number of bytes to read
    ///
    /// # Returns
    /// A vector containing the requested byte range, or an error if the range is out of bounds.
    async fn read(&self, offset: u64, length: u64) -> Result<Vec<u8>> {
        let end = offset + length;
        if end > self.file_length {
            return Err(anyhow!(
                "Read out of bounds: offset {} length {} exceeds file length {}",
                offset,
                length,
                self.file_length
            ));
        }

        let start = offset as usize;
        let end = end as usize;

        // Return a copy of the memory-mapped region.
        // While MMap allows zero-copy access via slices, the FileIO trait
        // returns a Vec<u8> to remain compatible with other implementations.
        Ok(self.mmap[start..end].to_vec())
    }

    /// Retrieves file metadata.
    async fn metadata(&self) -> Result<Metadata> {
        Ok(self.file.metadata()?)
    }

    /// Returns the length of the file in bytes.
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

    #[tokio::test]
    async fn test_mmap_read_basic() -> Result<()> {
        let temp_dir = TempDir::new("mmap_file_test")?;
        let file_path = temp_dir.path().join("test.txt");
        let data = b"hello memory mapped world";
        {
            let mut file = File::create(&file_path)?;
            file.write_all(data)?;
            file.flush()?;
        }

        let mmap_io = MMapFileIO::new(&file_path)?;

        // Read "hello"
        let buf = mmap_io.read(0, 5).await?;
        assert_eq!(buf, b"hello");

        // Read "world"
        let buf = mmap_io.read(20, 5).await?;
        assert_eq!(buf, b"world");

        assert_eq!(mmap_io.file_length().await?, data.len() as u64);

        Ok(())
    }

    #[tokio::test]
    async fn test_mmap_read_out_of_bounds() -> Result<()> {
        let temp_dir = TempDir::new("mmap_file_test")?;
        let file_path = temp_dir.path().join("test_short.txt");
        {
            let mut file = File::create(&file_path)?;
            file.write_all(b"short")?;
            file.flush()?;
        }

        let mmap_io = MMapFileIO::new(&file_path)?;

        // Try to read past EOF
        let result = mmap_io.read(0, 10).await;
        assert!(result.is_err());

        let result = mmap_io.read(5, 1).await;
        assert!(result.is_err());

        Ok(())
    }

    #[tokio::test]
    async fn test_mmap_empty_file() -> Result<()> {
        let temp_dir = TempDir::new("mmap_file_test")?;
        let file_path = temp_dir.path().join("empty.txt");
        File::create(&file_path)?;

        // Note: memmap2 might fail on mapping empty files depending on the OS.
        // However, some systems allow it. Let's see how MMapFileIO handles it.
        let mmap_io_result = MMapFileIO::new(&file_path);

        if let Ok(mmap_io) = mmap_io_result {
            assert_eq!(mmap_io.file_length().await?, 0);
            let buf = mmap_io.read(0, 0).await?;
            assert!(buf.is_empty());
        }

        Ok(())
    }
}
