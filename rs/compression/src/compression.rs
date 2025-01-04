use std::fs::File;
use std::io::BufWriter;

use anyhow::Result;

pub trait IntSeqEncoder {
    /// Creates an encoder
    fn new_encoder(universe: usize, num_elem: usize) -> Self
    where
        Self: Sized;

    /// Compresses a sorted slice of integers
    fn encode_batch(&mut self, slice: &[u64]) -> Result<()>;

    /// Compresses an u64 integer
    fn encode_value(&mut self, value: &u64) -> Result<()>;

    /// Returns the size of the encoded data (that would be written to disk)
    fn len(&self) -> usize;

    /// Writes to disk and returns number of bytes written (which can be just len(),
    /// or more if extra info is also required for decoding)
    fn write(&self, writer: &mut BufWriter<&mut File>) -> Result<usize>;
}

pub trait IntSeqDecoder {
    type IteratorType<'a>: Iterator<Item = Self::Item>;
    type Item;

    /// Creates a decoder
    fn new_decoder(byte_slice: &[u8]) -> Self
    where
        Self: Sized;

    /// Creates an iterator that iterates the encoded data and decodes one element at a time on the
    /// fly
    fn get_iterator<'a>(&self, encoded_data: &'a [u8]) -> Self::IteratorType<'a>;

    /// Returns the number of elements in the sequence
    fn num_elem(&self) -> usize;
}
