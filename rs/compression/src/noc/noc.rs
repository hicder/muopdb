use std::fs::File;
use std::io::{BufWriter, Write};
use std::ptr::NonNull;

use anyhow::{anyhow, Result};
use utils::io::wrap_write;
use utils::mem::get_ith_val_from_raw_ptr;

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
    encoded_data_ptr: NonNull<u64>,
}

impl IntSeqDecoder for PlainDecoder {
    type IteratorType = PlainDecodingIterator;
    type Item = u64;

    fn new_decoder(encoded_data: &[u8]) -> Self {
        let encoded_data_ptr = NonNull::new(encoded_data.as_ptr() as *mut u64)
            .expect("Encoded data pointer should not be null");
        Self {
            size: encoded_data.len(),
            encoded_data_ptr,
        }
    }

    fn get_iterator(&self) -> Self::IteratorType {
        PlainDecodingIterator {
            num_elem: self.num_elem(),
            cur_index: 0,
            encoded_data_ptr: self.encoded_data_ptr,
        }
    }

    fn num_elem(&self) -> usize {
        self.size / std::mem::size_of::<Self::Item>()
    }
}

pub struct PlainDecodingIterator {
    num_elem: usize,
    cur_index: usize,
    encoded_data_ptr: NonNull<u64>,
}

impl Iterator for PlainDecodingIterator {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cur_index < self.num_elem {
            let value = get_ith_val_from_raw_ptr(self.encoded_data_ptr.as_ptr(), self.cur_index);
            self.cur_index += 1;
            Some(value)
        } else {
            None
        }
    }
}
