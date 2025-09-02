use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};

use anyhow::Result;
use byteorder::{LittleEndian, ReadBytesExt};
use log::{info, warn};
use rkyv::util::AlignedVec;
use utils::mem::{transmute_slice_to_u8, transmute_u8_to_val_unaligned};

use super::entry::{WalEntry, WalOpType};

const VERSION_1: &[u8] = b"version1";

/// A single WAL file. This file will have 3 sections:
/// | version1 | start_seq_no | data |
/// | 8 bytes  | 8 bytes      | ...  |
///
/// Each data entry will have the following format (n is number of doc_ids, m is number of user_ids):
/// | length   | n       | m       | doc_ids      | user_ids     | data                       | op_type |
/// | 4 bytes  | 8 bytes | 8 bytes | 16 bytes * n | 16 bytes * m | 4 bytes * n * num_features | 1 byte  |
///
#[allow(unused)]
pub struct WalFile {
    file: File,

    // metadata: Metadata,
    path: String,

    // The start sequence number of the file
    start_seq_no: i64,

    // The number of entries in the file
    num_entries: u32,
}

impl WalFile {
    pub fn create(path: &str, start_seq_no: i64) -> Result<Self> {
        // If the file does not exist, create it
        let mut file = OpenOptions::new().create(true).append(true).open(path)?;
        // Write "version1" to the file
        file.write_all(b"version1")?;
        // Write the start sequence number to the file
        file.write_all(&start_seq_no.to_le_bytes())?;
        // Flush
        file.flush()?;

        Ok(Self {
            file,
            path: path.to_string(),
            start_seq_no,
            num_entries: 0,
        })
    }

    pub fn open(path: &str) -> Result<Self> {
        let mut file = OpenOptions::new().read(true).write(true).open(path)?;

        // Interestingly, you should always sync the file on open.
        file.sync_all()?;

        file.seek(SeekFrom::Start(VERSION_1.len() as u64))?;

        // Read the start sequence number
        let mut buf = vec![0; 8];
        file.read_exact(&mut buf)?;
        let start_seq_no = transmute_u8_to_val_unaligned::<i64>(&buf);

        // This will traverse the entire file.
        let num_entries = Self::read_num_entries_from_file(&mut file)?;

        info!(
            "Opened WAL file {} with start seq no {} and {} entries",
            path, start_seq_no, num_entries
        );

        // Reopen with append only for writing.
        file = OpenOptions::new().append(true).open(path)?;
        info!(
            "Opened WAL file {} with start seq no {} and {} entries",
            path, start_seq_no, num_entries
        );
        Ok(Self {
            file,
            path: path.to_string(),
            start_seq_no,
            num_entries,
        })
    }

    pub fn append(
        &mut self,
        doc_ids: &[u128],
        user_ids: &[u128],
        data: &[f32],
        op_type: WalOpType,
    ) -> Result<u64> {
        let len = (8 + 8 + doc_ids.len() * 16 + user_ids.len() * 16 + data.len() * 4 + 1) as u32;
        self.file.write_all(&len.to_le_bytes())?;
        self.file.write_all(&(doc_ids.len() as u64).to_le_bytes())?;
        self.file
            .write_all(&(user_ids.len() as u64).to_le_bytes())?;
        self.file.write_all(transmute_slice_to_u8(doc_ids))?;
        self.file.write_all(transmute_slice_to_u8(user_ids))?;
        self.file.write_all(transmute_slice_to_u8(data))?;
        self.file.write_all(&(op_type as u8).to_le_bytes())?;
        self.file.flush()?;
        self.file.sync_data()?;

        // Increment the number of entries in the file
        self.num_entries += 1;
        Ok((self.start_seq_no + self.num_entries as i64) as u64)
    }

    pub fn append_raw(&mut self, data: &[u8]) -> Result<u64> {
        // Write the length first as u32
        let length = data.len() as u32;
        self.file.write_all(&length.to_le_bytes())?;
        self.file.write_all(data)?;
        self.file.flush()?;
        self.file.sync_data()?;

        // Increment the number of entries in the file
        self.num_entries += 1;
        Ok((self.start_seq_no + self.num_entries as i64) as u64)
    }

    pub fn get_iterator(&self) -> Result<WalFileIterator> {
        let f = File::open(&self.path)?;
        WalFileIterator::new(f, self.num_entries)
    }

    pub fn get_num_entries(&self) -> u32 {
        self.num_entries
    }

    pub fn read_num_entries_from_file(file: &mut File) -> Result<u32> {
        let mut num_entries = 0;

        // Start reading from offset 16
        let mut offset = 16;
        let file_len = file.metadata()?.len();

        while offset < file_len {
            num_entries += 1;
            file.seek(SeekFrom::Start(offset))?;

            // Read the length (4 bytes)
            let length = file.read_u32::<LittleEndian>()?;
            offset = offset + 4 + length as u64;
        }

        if offset != file_len {
            warn!("WAL file is not complete");
        }

        Ok(num_entries)
    }

    pub fn get_file_size(&self) -> Result<u64> {
        let metadata = self.file.metadata()?;
        Ok(metadata.len())
    }

    pub fn get_start_seq_no(&self) -> i64 {
        self.start_seq_no
    }

    pub fn get_last_seq_no(&self) -> i64 {
        self.start_seq_no + self.num_entries as i64
    }

    pub fn get_path(&self) -> &str {
        &self.path
    }

    pub fn get_wal_id(&self) -> u32 {
        self.path.split(".").last().unwrap().parse::<u32>().unwrap()
    }
}

#[allow(unused)]
pub struct WalFileIterator {
    file: File,
    file_size: u64,
    offset: u64,
    current_seq_no: i64,
    num_entries: u32,
}

impl WalFileIterator {
    pub fn new(file: File, num_entries: u32) -> Result<Self> {
        let mut file = file;
        let file_size = file.metadata()?.len();
        file.seek(SeekFrom::Start(VERSION_1.len() as u64))?;

        let mut buf = vec![0; 8];
        file.read_exact(&mut buf)?;
        let start_seq_no = transmute_u8_to_val_unaligned::<i64>(&buf);

        Ok(Self {
            file,
            file_size,
            offset: VERSION_1.len() as u64 + 8,
            current_seq_no: start_seq_no,
            num_entries,
        })
    }

    /// Get the last sequence number in the file
    pub fn last_seq_no(&self) -> i64 {
        self.current_seq_no + self.num_entries as i64
    }

    /// Skip to the the sequence number that is less than or equal to the given sequence number
    pub fn skip_to(&mut self, seq_no: i64) -> Result<i64> {
        let mut idx = seq_no - self.current_seq_no - 1;
        while idx > 0 {
            if self.offset >= self.file_size {
                break;
            }

            self.file.seek(SeekFrom::Start(self.offset))?;
            let length = self.file.read_u32::<LittleEndian>()?;
            self.offset += length as u64 + 4;
            self.current_seq_no += 1;
            idx -= 1;
        }
        Ok(self.current_seq_no)
    }
}

impl Iterator for WalFileIterator {
    type Item = Result<WalEntry>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.offset >= self.file_size {
            return None;
        }

        self.file.seek(SeekFrom::Start(self.offset)).unwrap();

        let length = self.file.read_u32::<LittleEndian>().unwrap();
        self.offset += 4;
        if self.offset + length as u64 > self.file_size {
            return None;
        }
        self.offset += length as u64;

        let mut buffer = AlignedVec::<16>::with_capacity(length as usize);
        buffer.resize(length as usize, 0);

        self.file.read_exact(buffer.as_mut_slice()).unwrap();
        self.current_seq_no += 1;
        let seq_no = self.current_seq_no;

        Some(Ok(WalEntry {
            buffer,
            seq_no: seq_no as u64,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wal() {
        let tmp_dir = tempdir::TempDir::new("wal_file_test").unwrap();
        let mut wal_file =
            WalFile::create(tmp_dir.path().join("test.wal").to_str().unwrap(), -1).unwrap();

        // Append some data
        wal_file.append_raw(b"hello").unwrap();
        assert_eq!(wal_file.get_num_entries(), 1);

        // Append some more data
        wal_file.append_raw(b"world").unwrap();
        assert_eq!(wal_file.get_num_entries(), 2);
    }

    #[test]
    fn test_get_entry() {
        let tmp_dir = tempdir::TempDir::new("wal_file_test").unwrap();
        let mut wal_file =
            WalFile::create(tmp_dir.path().join("test.wal").to_str().unwrap(), 100).unwrap();

        // Append some data
        wal_file.append_raw(b"hello").unwrap();
        wal_file.append_raw(b"world").unwrap();

        let mut iter = wal_file.get_iterator().unwrap();
        let start_seq_no = iter.current_seq_no;
        assert_eq!(start_seq_no, 100);

        let entry = iter.next().unwrap().unwrap();
        assert_eq!(entry.buffer.as_slice(), b"hello");
        assert_eq!(entry.seq_no, 101);

        let entry = iter.next().unwrap().unwrap();
        assert_eq!(entry.buffer.as_slice(), b"world");
        assert_eq!(entry.seq_no, 102);

        let entry = iter.next();
        assert!(entry.is_none());
    }

    #[test]
    fn test_skip_to() {
        let tmp_dir = tempdir::TempDir::new("wal_file_test").unwrap();
        let mut wal_file =
            WalFile::create(tmp_dir.path().join("test.wal").to_str().unwrap(), -1).unwrap();

        // Insert hello_x for x = 0, 1, 2, 3, 4
        for x in 0..5 {
            wal_file
                .append_raw(format!("hello_{}", x).as_bytes())
                .unwrap();
        }

        let mut iter = wal_file.get_iterator().unwrap();
        assert_eq!(iter.last_seq_no(), 4);

        iter.skip_to(2).unwrap();
        let entry = iter.next().unwrap().unwrap();
        assert_eq!(entry.buffer.as_slice(), b"hello_2");
        assert_eq!(entry.seq_no, 2);

        let entry = iter.next().unwrap().unwrap();
        assert_eq!(entry.buffer.as_slice(), b"hello_3");
        assert_eq!(entry.seq_no, 3);

        let seq_no = iter.skip_to(10).unwrap();
        assert_eq!(seq_no, 4);
        assert!(iter.next().is_none());
    }
}
