use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};

/// Convenient wrapper for going from io::Result<usize> to Result<usize, String>
pub fn wrap_write(writer: &mut BufWriter<&mut File>, buf: &[u8]) -> anyhow::Result<usize> {
    anyhow::Ok(writer.write(buf)?)
}

/// Read file and append to the writer
pub fn append_file_to_writer(
    path: &str,
    writer: &mut BufWriter<&mut File>,
) -> anyhow::Result<usize> {
    let input_file = File::open(path).unwrap();
    let mut buffer_reader = BufReader::new(&input_file);
    let mut buffer: [u8; 4096] = [0; 4096];
    let mut written = 0;
    loop {
        let read = buffer_reader.read(&mut buffer).unwrap();
        written += wrap_write(writer, &buffer[0..read])?;
        if read < 4096 {
            break;
        }
    }
    Ok(written)
}

pub fn get_latest_version(config_path: &str) -> u64 {
    // List all files in the directory
    let mut latest_version = 0;
    for entry in std::fs::read_dir(config_path).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        let filename = path.file_name().unwrap().to_str().unwrap();
        if filename.starts_with("version_") {
            let version = filename.split("_").last().unwrap();
            let version = version.parse::<u64>().unwrap();
            if version > latest_version {
                latest_version = version;
            }
        }
    }
    latest_version
}
