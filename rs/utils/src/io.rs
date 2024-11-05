use std::fs::File;
use std::io::{BufWriter, Write};

/// Convenient wrapper for going from io::Result<usize> to Result<usize, String>
pub fn wrap_write(writer: &mut BufWriter<&mut File>, buf: &[u8]) -> anyhow::Result<usize> {
    anyhow::Ok(writer.write(buf)?)
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
