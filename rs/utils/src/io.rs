use std::fs::File;
use std::io::{BufWriter, Write};

/// Convenient wrapper for going from io::Result<usize> to Result<usize, String>
pub fn wrap_write(writer: &mut BufWriter<&mut File>, buf: &[u8]) -> anyhow::Result<usize> {
    anyhow::Ok(writer.write(buf)?)
}
