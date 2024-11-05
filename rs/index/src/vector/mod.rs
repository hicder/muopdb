use std::fs::File;
use std::io::BufWriter;

use anyhow::Result;
use num_traits::ops::bytes::ToBytes;

pub mod file;
pub mod fixed_file;

/// Trait that defines the interface for vector storage
/// This storage owns the actual vector, and will return a reference to it
pub trait VectorStorage<T: ToBytes + Clone> {
    fn get(&self, id: u32) -> Option<&[T]>;
    fn append(&mut self, vector: &[T]) -> Result<()>;

    // Number of vectors in the storage
    fn len(&self) -> usize;

    // Return number of bytes written.
    fn write(&self, writer: &mut BufWriter<&mut File>) -> Result<usize>;
}
