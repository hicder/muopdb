use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;

use anyhow::Result;

pub struct Scratch {
    file: File,
}

impl Scratch {
    pub fn new(file_path: &Path) -> std::io::Result<Self> {
        // Append only
        let file = OpenOptions::new()
            .append(true)
            .create(true)
            .open(file_path)?;

        Ok(Self { file })
    }

    pub fn write(&mut self, point_id: u32, term_id: u64) -> Result<()> {
        let mut buffer =
            Vec::with_capacity(std::mem::size_of::<u32>() + std::mem::size_of::<u64>());
        buffer.extend_from_slice(&point_id.to_le_bytes());
        buffer.extend_from_slice(&term_id.to_le_bytes());
        self.file.write_all(&buffer)?;

        Ok(())
    }
}
