use std::fs::File;
use std::io::{BufWriter, Write};

use anyhow::Result;
use utils::io::wrap_write;

use crate::compression::IntSeqEncoder;

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
