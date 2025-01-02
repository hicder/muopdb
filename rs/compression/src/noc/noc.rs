use std::fs::File;
use std::io::{BufWriter, Write};

use anyhow::Result;
use utils::io::wrap_write;

use crate::compression::IntSeqEncoder;

pub struct PlainEncoder {
    size: usize,
    sequence: Vec<u64>,
}

impl PlainEncoder {
    pub fn new(size: usize) -> Self {
        Self {
            size,
            sequence: Vec::new(),
        }
    }
}

impl IntSeqEncoder for PlainEncoder {
    fn new_encoder(_universe: Option<usize>, size: usize) -> Self {
        Self::new(size)
    }

    fn encode(&mut self, values: &[u64]) -> Result<()> {
        self.sequence = values.to_vec();
        Ok(())
    }

    fn len(&self) -> usize {
        self.size
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
