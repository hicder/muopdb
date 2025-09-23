use std::fs::File;
use std::io::{BufWriter, Write};

use anyhow::{anyhow, Result};
use utils::io::wrap_write;

use crate::compression::{CompressionInt, IntSeqDecoder, IntSeqEncoder};

pub struct PlainEncoder<T: CompressionInt = u64> {
    num_elem: usize,
    sequence: Vec<T>,
}

impl<T: CompressionInt> PlainEncoder<T> {
    pub fn new(num_elem: usize) -> Self {
        Self {
            num_elem,
            sequence: Vec::new(),
        }
    }
}

impl<T: CompressionInt> IntSeqEncoder<T> for PlainEncoder<T> {
    fn new_encoder(_universe: T, num_elem: usize) -> Self {
        Self::new(num_elem)
    }

    fn encode_value(&mut self, value: &T) -> Result<()> {
        self.sequence.push(*value);
        Ok(())
    }

    fn encode_batch(&mut self, slice: &[T]) -> Result<()> {
        self.sequence.extend(slice);
        Ok(())
    }

    fn len(&self) -> usize {
        self.num_elem * std::mem::size_of::<T>()
    }

    fn write(&self, writer: &mut BufWriter<&mut File>) -> Result<usize> {
        let mut total_bytes_written = 0;

        for &val in self.sequence.iter() {
            // Convert the value to bytes for writing
            let bytes = val.to_le_bytes();
            total_bytes_written += wrap_write(writer, bytes.as_ref())?;
        }

        if total_bytes_written != self.len() {
            return Err(anyhow!(
                "Expected to write {} bytes but wrote {} bytes",
                self.len(),
                total_bytes_written
            ));
        }
        writer.flush()?;

        Ok(total_bytes_written)
    }
}

pub struct PlainDecoder<T: CompressionInt = u64> {
    size: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: CompressionInt> PlainDecoder<T> {
    pub fn num_elem(&self) -> usize {
        self.size / std::mem::size_of::<T>()
    }
}

impl<T: CompressionInt> IntSeqDecoder<T> for PlainDecoder<T> {
    type IteratorType<'a>
        = PlainDecodingIterator<'a, T>
    where
        T: 'a;

    fn new_decoder(byte_slice: &[u8]) -> Result<Self> {
        Ok(Self {
            size: byte_slice.len(),
            _phantom: std::marker::PhantomData,
        })
    }

    fn get_iterator<'a>(&self, byte_slice: &'a [u8]) -> Self::IteratorType<'a>
    where
        T: 'a,
    {
        PlainDecodingIterator {
            num_elem: self.num_elem(),
            cur_index: 0,
            encoded_data: utils::mem::transmute_u8_to_slice(byte_slice),
            _phantom: std::marker::PhantomData,
        }
    }
}

pub struct PlainDecodingIterator<'a, T: CompressionInt> {
    num_elem: usize,
    cur_index: usize,
    encoded_data: &'a [u8], // Keep as u8 slice for generic handling
    _phantom: std::marker::PhantomData<T>,
}

impl<'a, T: CompressionInt> PlainDecodingIterator<'a, T> {
    fn get_value_at_index(&self, index: usize) -> T {
        let type_size = std::mem::size_of::<T>();
        let start = index * type_size;
        let end = start + type_size;

        T::from_le_bytes(&self.encoded_data[start..end]).unwrap_or_else(|_| T::zero())
    }
}

impl<'a, T: CompressionInt> Iterator for PlainDecodingIterator<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cur_index < self.num_elem {
            let value = self.get_value_at_index(self.cur_index);
            self.cur_index += 1;
            Some(value)
        } else {
            None
        }
    }
}
