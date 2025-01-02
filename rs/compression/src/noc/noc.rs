use std::fs::File;
use std::io::{BufWriter, Write};

use anyhow::Result;
use memmap2::Mmap;
use utils::io::wrap_write;
use utils::mem::transmute_u8_to_slice;

use crate::compression::{IntSeqDecoderIterator, IntSeqEncoder};

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
    fn new_encoder(_universe: Option<usize>, num_elem: usize) -> Self {
        Self::new(num_elem)
    }

    fn encode(&mut self, values: &[u64]) -> Result<()> {
        self.sequence = values.to_vec();
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

        writer.flush()?;

        Ok(total_bytes_written)
    }
}

pub struct PlainDecoderIterator<'a> {
    size: usize,
    cur_index: usize,
    encoded_data_ptr: &'a [u64],
}

impl<'a> IntSeqDecoderIterator<'a> for PlainDecoderIterator<'a> {
    fn new_decoder(mmap: &'a Mmap, offset: usize, size: usize) -> Self {
        let slice = &mmap[offset..offset + size * size_of::<u64>()];
        let encoded_data = transmute_u8_to_slice::<u64>(slice);
        Self {
            size,
            cur_index: 0,
            encoded_data_ptr: encoded_data,
        }
    }

    fn len(&self) -> usize {
        self.size
    }
}

impl Iterator for PlainDecoderIterator<'_> {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cur_index < self.size {
            Some(self.encoded_data_ptr[self.cur_index])
        } else {
            None
        }
    }
}
