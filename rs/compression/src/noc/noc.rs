use std::fs::File;
use std::io::{BufWriter, Write};

use anyhow::{anyhow, Result};
use utils::io::wrap_write;

use crate::compression::{IntSeqDecoder, IntSeqEncoder};

pub struct PlainEncoder {
    num_elem: usize,
    sequence: Vec<u64>,
}

impl PlainEncoder {
    pub fn new(num_elem: usize) -> Self {
        Self {
            num_elem,
            sequence: Vec::new(),
        }
    }
}

impl IntSeqEncoder for PlainEncoder {
    fn new_encoder(_universe: usize, num_elem: usize) -> Self {
        Self::new(num_elem)
    }

    fn encode_value(&mut self, value: &u64) -> Result<()> {
        self.sequence.push(*value);
        Ok(())
    }

    fn encode_batch(&mut self, slice: &[u64]) -> Result<()> {
        self.sequence.extend(slice);
        Ok(())
    }

    fn len(&self) -> usize {
        self.num_elem * std::mem::size_of::<u64>()
    }

    fn write(&self, writer: &mut BufWriter<&mut File>) -> Result<usize> {
        let mut total_bytes_written = 0;

        for &val in self.sequence.iter() {
            total_bytes_written += wrap_write(writer, &val.to_le_bytes())?;
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

pub struct PlainDecoder {
    size: usize,
}

impl PlainDecoder {
    pub fn num_elem(&self) -> usize {
        self.size / std::mem::size_of::<<PlainDecoder as IntSeqDecoder>::Item>()
    }
}

impl IntSeqDecoder for PlainDecoder {
    type IteratorType<'a> = PlainDecodingIterator<'a>;
    type Item = u64;

    fn new_decoder(encoded_data: &[u8]) -> Result<Self> {
        Ok(Self {
            size: encoded_data.len(),
        })
    }

    fn get_iterator<'a>(&self, encoded_data: &'a [u8]) -> PlainDecodingIterator<'a> {
        PlainDecodingIterator {
            num_elem: self.num_elem(),
            cur_index: 0,
            encoded_data_ptr: utils::mem::transmute_u8_to_slice(encoded_data),
        }
    }
}

pub struct PlainDecodingIterator<'a> {
    num_elem: usize,
    cur_index: usize,
    encoded_data_ptr: &'a [u64],
}

impl<'a> Iterator for PlainDecodingIterator<'a> {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cur_index < self.num_elem {
            self.cur_index += 1;
            Some(self.encoded_data_ptr[self.cur_index - 1])
        } else {
            None
        }
    }
}
