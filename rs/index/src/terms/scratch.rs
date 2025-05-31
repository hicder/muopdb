use std::fs::{File, OpenOptions};
use std::io::Write;

use anyhow::Result;

pub struct Scratch {
    file: File,
}

impl Scratch {
    pub fn new(file_path: &str) -> Self {
        // Append only
        let file = OpenOptions::new()
            .append(true)
            .create(true)
            .open(file_path)
            .unwrap();

        Self { file }
    }

    pub fn write(&mut self, doc_id: u64, term_id: u64) -> Result<()> {
        let mut buffer = Vec::with_capacity(16);
        buffer.extend_from_slice(&doc_id.to_le_bytes());
        buffer.extend_from_slice(&term_id.to_le_bytes());
        self.file.write_all(&buffer)?;

        Ok(())
    }
}
