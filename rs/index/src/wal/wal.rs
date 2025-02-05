use std::collections::VecDeque;

use anyhow::Result;
use log::info;

use crate::wal::entry::WalOpType;
use crate::wal::file::WalFile;
use crate::wal::file::WalFileIterator;

#[allow(unused)]
pub struct Wal {
    directory: String,
    files: VecDeque<WalFile>,

    // The size of the wal file.
    max_file_size: u64,
}

impl Wal {
    pub fn open(directory: &str, max_file_size: u64) -> Result<Self> {
        if !std::path::Path::new(directory).exists() {
            info!("Wal directory {} does not exist, creating it", directory);
            std::fs::create_dir_all(directory)?;
        }

        let mut files = VecDeque::new();

        // Get all files in the directory. Each file will have the name of wal.<file_id>
        let mut file_paths = std::fs::read_dir(directory)?
            .map(|file_path| file_path.unwrap().path().to_str().unwrap().to_owned())
            .collect::<Vec<_>>();
        file_paths.sort();

        if file_paths.is_empty() {
            info!("No wal files found, creating a new one");

            // Create a new wal file
            let file_path = format!("{}/wal.0", directory);
            let wal = WalFile::create(&file_path, -1)?;
            files.push_back(wal);
        } else {
            for file_path in file_paths {
                files.push_back(WalFile::open(&file_path)?);
            }
        }

        Ok(Self {
            directory: directory.to_string(),
            files,
            max_file_size,
        })
    }

    pub fn get_iterators(&self) -> Vec<WalFileIterator> {
        self.files
            .iter()
            .map(|file| file.get_iterator().unwrap())
            .collect()
    }

    pub fn append(
        &mut self,
        doc_ids: &[u128],
        user_ids: &[u128],
        data: &[f32],
        op_type: WalOpType,
    ) -> Result<u64> {
        let last_file = self.files.back().unwrap();
        if last_file.get_file_size()? >= self.max_file_size {
            let seq_no = last_file.get_start_seq_no() + last_file.get_num_entries() as i64;
            let file_path = format!("{}/wal.{}", self.directory, self.files.len());
            let wal = WalFile::create(&file_path, seq_no)?;
            self.files.push_back(wal);
        }
        self.files
            .back_mut()
            .unwrap()
            .append(doc_ids, user_ids, data, op_type)
    }

    /// Append a new entry to the wal. If the last file is full, create a new file.
    pub fn append_raw(&mut self, data: &[u8]) -> Result<u64> {
        let last_file = self.files.back().unwrap();
        if last_file.get_file_size()? >= self.max_file_size {
            let seq_no = last_file.get_start_seq_no() + last_file.get_num_entries() as i64;
            let file_path = format!("{}/wal.{}", self.directory, self.files.len());
            let wal = WalFile::create(&file_path, seq_no)?;
            self.files.push_back(wal);
        }
        self.files.back_mut().unwrap().append_raw(data)
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
            let mut wal = WalFile::create(&file_path.to_str().unwrap(), i * 10 - 1).unwrap();
            for j in 0..10 {
                wal.append_raw(&format!("hello_{}", j).as_bytes()).unwrap();
            }
        }

        let wal = Wal::open(dir.to_str().unwrap(), 1024).unwrap();
        let iterators = wal.get_iterators();
        for (i, iterator) in iterators.iter().enumerate() {
            assert_eq!(iterator.last_seq_no(), (i as i64 + 1) * 10 - 1);
        }
    }

    #[test]
    fn test_wal_open_empty_dir() {
        let tmp_dir = tempdir::TempDir::new("wal_test").unwrap();
        let dir = tmp_dir.path();
        let wal = Wal::open(dir.to_str().unwrap(), 1024).unwrap();
        assert_eq!(wal.files.len(), 1);
        assert_eq!(wal.files[0].get_num_entries(), 0);
    }

    #[test]
    fn test_wal_append_raw() {
        let tmp_dir = tempdir::TempDir::new("wal_test").unwrap();
        let dir = tmp_dir.path();
        let mut wal = Wal::open(dir.to_str().unwrap(), 1024).unwrap();
        for i in 0..100 {
            let seq_no = wal.append_raw(&format!("hello{}", i).as_bytes()).unwrap();
            assert_eq!(seq_no, i as u64);
        }
        assert_eq!(wal.files.len(), 2);

        // Check that all entries are in the correct order
        let mut start = 0;
        let iterators = wal.get_iterators();
        for it in iterators {
            for i in it {
                assert_eq!(i.unwrap().seq_no, start);
                start += 1;
            }
        }

        assert_eq!(start, 100);
    }

    #[test]
    fn test_wal_append() {
        let tmp_dir = tempdir::TempDir::new("wal_test").unwrap();
        let dir = tmp_dir.path();
        let mut wal = Wal::open(dir.to_str().unwrap(), 1024).unwrap();
        for i in 0..5 {
            let data = vec![i as f32; 10];
            let seq_no = wal
                .append(&vec![i as u128], &vec![i as u128], &data, WalOpType::Insert)
                .unwrap();
            assert_eq!(seq_no, i as u64);
        }
        assert_eq!(wal.files.len(), 1);

        let mut iterators = wal.get_iterators();
        let mut last_it = iterators.pop().unwrap();
        for i in 0..5 {
            let entry = last_it.next().unwrap().unwrap();
            assert_eq!(entry.seq_no, i as u64);
            let decoded = entry.decode(10);
            assert_eq!(decoded.doc_ids, vec![i as u128]);
            assert_eq!(decoded.user_ids, vec![i as u128]);
            assert_eq!(decoded.data, vec![i as f32; 10]);
            assert_eq!(decoded.op_type, WalOpType::Insert);
        }
    }

    #[test]
    fn test_wal_append_delete() {
        let tmp_dir = tempdir::TempDir::new("wal_test").unwrap();
        let dir = tmp_dir.path();
        let mut wal = Wal::open(dir.to_str().unwrap(), 1024).unwrap();
        for i in 0..5 {
            let data = vec![i as f32; 10];
            let seq_no = wal
                .append(&vec![i as u128], &vec![i as u128], &data, WalOpType::Delete)
                .unwrap();
            assert_eq!(seq_no, i as u64);
        }
        assert_eq!(wal.files.len(), 1);

        let mut iterators = wal.get_iterators();
        let mut last_it = iterators.pop().unwrap();
        for i in 0..5 {
            let entry = last_it.next().unwrap().unwrap();
            assert_eq!(entry.seq_no, i as u64);
            let decoded = entry.decode(10);
            assert_eq!(decoded.doc_ids, vec![i as u128]);
            assert_eq!(decoded.user_ids, vec![i as u128]);
            assert_eq!(decoded.data, vec![i as f32; 10]);
            assert_eq!(decoded.op_type, WalOpType::Delete);
        }
    }
}
