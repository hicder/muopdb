use std::{
    fs::File,
    io::{BufWriter, Write},
};

/// Convenient wrapper for going from io::Result<usize> to Result<usize, String>
pub fn wrap_write(writer: &mut BufWriter<&mut File>, buf: &[u8]) -> Result<usize, String> {
    match writer.write(buf) {
        Ok(len) => Ok(len),
        Err(e) => Err(e.to_string()),
    }
}
