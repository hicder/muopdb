use std::fs::File;
use std::io::BufWriter;

use anyhow::Result;

pub trait IntSeqEncoder {
    /// Creates an encoder
    fn new_encoder(universe: Option<usize>, size: usize) -> Self
    where
        Self: Sized;

    /// Compresses a sorted slice of integers
    fn encode(&mut self, values: &[u64]) -> Result<()>;

    /// Returns the number of elements in the sequence
    fn len(&self) -> usize;

    /// Writes to disk and return number of bytes written.
    fn write(&self, writer: &mut BufWriter<&mut File>) -> Result<usize>;
}
