use std::fs::{read_dir, File};
use std::io::{BufReader, BufWriter, Read, Write};

use anyhow::{anyhow, Result};

/// Convenient wrapper for going from io::Result<usize> to Result<usize, String>
pub fn wrap_write<W: Write>(writer: &mut W, buf: &[u8]) -> Result<usize> {
    anyhow::Ok(writer.write(buf)?)
}

/// Read file and append to the writer
pub fn append_file_to_writer(path: &str, writer: &mut BufWriter<&mut File>) -> Result<usize> {
    let input_file = File::open(path)?;
    let mut buffer_reader = BufReader::new(&input_file);
    let mut buffer: [u8; 4096] = [0; 4096];
    let mut written = 0;
    loop {
        let read = buffer_reader.read(&mut buffer)?;
        written += wrap_write(writer, &buffer[0..read])?;
        if read < 4096 {
            break;
        }
    }
    Ok(written)
}

pub fn get_latest_version(config_path: &str) -> Result<u64> {
    // List all files in the directory
    let mut latest_version = 0;
    for entry in read_dir(config_path)? {
        let path = entry?.path();
        let filename = path
            .file_name()
            .and_then(|os_str| os_str.to_str())
            .ok_or_else(|| anyhow!("Invalid filename"))?;
        if filename.starts_with("version_") {
            let version = filename
                .split('_')
                .next_back()
                .ok_or_else(|| anyhow!("Invalid version format"))?
                .parse::<u64>()?;
            latest_version = latest_version.max(version);
        }
    }
    Ok(latest_version)
}

pub fn write_pad(
    written: usize,
    writer: &mut BufWriter<&mut File>,
    alignment: usize,
) -> Result<usize> {
    let mut padded = 0;
    let padding = alignment - (written % alignment);
    if padding != alignment {
        let padding_buffer = vec![0; padding];
        padded += wrap_write(writer, &padding_buffer)?;
    }
    Ok(padded)
}

// Test
#[cfg(test)]
mod tests {
    use std::fs::{read, write};

    use tempdir::TempDir;

    use super::*;

    #[test]
    fn test_append_file_to_writer() -> Result<()> {
        let temp_dir = TempDir::new("append_file_to_writer_test")?;
        let base_directory = temp_dir
            .path()
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("Failed to convert temp directory path to string"))?
            .to_string();

        // Create a test file
        let test_content = b"Hello, World!";
        write(format!("{}/test_file", base_directory), test_content)?;

        // Create a target file
        let mut target_file = File::create(format!("{}/target_file", base_directory))?;
        let mut buffer_writer = BufWriter::new(&mut target_file);

        let written =
            append_file_to_writer(&format!("{}/test_file", base_directory), &mut buffer_writer)?;

        assert_eq!(written, test_content.len());

        buffer_writer.flush()?;
        drop(buffer_writer);

        let content = read(format!("{}/target_file", base_directory))?;
        assert_eq!(content, test_content);

        Ok(())
    }

    #[test]
    fn test_get_latest_version() -> Result<()> {
        let temp_dir = TempDir::new("get_latest_version_test")?;
        let base_directory = temp_dir
            .path()
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("Failed to convert temp directory path to string"))?
            .to_string();

        // Create three version files
        write(format!("{}/version_1", base_directory), "version 1")?;
        write(format!("{}/version_2", base_directory), "version 2")?;
        write(format!("{}/version_3", base_directory), "version 3")?;

        let latest_version = get_latest_version(&base_directory)?;
        assert_eq!(latest_version, 3);

        Ok(())
    }
}
