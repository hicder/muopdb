use anyhow::Result;

use super::file::WalFileIterator;
use crate::wal::file::WalFile;

#[allow(unused)]
pub struct Wal {
    directory: String,
    files: Vec<WalFile>,
}

impl Wal {
    pub fn open(directory: &str) -> Result<Self> {
        let mut files = Vec::new();

        // Get all files in the directory. Each file will have the name of wal.<file_id>
        let mut file_paths = std::fs::read_dir(directory)?
            .map(|file_path| file_path.unwrap().path().to_str().unwrap().to_owned())
            .collect::<Vec<_>>();
        file_paths.sort();

        if file_paths.is_empty() {
            // Create a new wal file
            let file_path = format!("{}/wal.0", directory);
            let wal = WalFile::create(&file_path, 0)?;
            files.push(wal);
        } else {
            for file_path in file_paths {
                files.push(WalFile::open(&file_path)?);
            }
        }

        Ok(Self {
            directory: directory.to_string(),
            files,
        })
    }

    pub fn get_iterators(&self) -> Vec<WalFileIterator> {
        self.files
            .iter()
            .map(|file| file.get_iterator().unwrap())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wal() {
        // Create 3 wal files. First one has seq_no from 0 to 9, second one has seq_no from 10 to 19, and the third one has seq_no from 20 to 29.
        let tmp_dir = tempdir::TempDir::new("wal_test").unwrap();
        let dir = tmp_dir.path();
        for i in 0..3 {
            let file_path = dir.join(format!("wal.{}", i));
            let mut wal = WalFile::create(&file_path.to_str().unwrap(), i * 10).unwrap();
            for j in 0..10 {
                wal.append(&format!("hello_{}", j).as_bytes()).unwrap();
            }
        }

        let wal = Wal::open(dir.to_str().unwrap()).unwrap();
        let iterators = wal.get_iterators();
        for (i, iterator) in iterators.iter().enumerate() {
            assert_eq!(iterator.last_seq_no(), (i as u64 + 1) * 10 - 1);
        }
    }

    #[test]
    fn test_wal_open_empty_dir() {
        let tmp_dir = tempdir::TempDir::new("wal_test").unwrap();
        let dir = tmp_dir.path();
        let wal = Wal::open(dir.to_str().unwrap()).unwrap();
        assert_eq!(wal.files.len(), 1);
        assert_eq!(wal.files[0].get_num_entries(), 0);
    }
}
