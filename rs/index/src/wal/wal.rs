use std::collections::VecDeque;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;

use anyhow::Result;
use log::info;
use proto::muopdb::DocumentAttribute;
use tokio::sync::watch;
use utils::file_io::env::Env;

use crate::wal::entry::WalOpType;
use crate::wal::file::{WalFile, WalFileIterator};

#[allow(unused)]
pub struct Wal {
    directory: String,
    files: VecDeque<WalFile>,
    env: Arc<Box<dyn Env>>,

    // The size of the wal file.
    max_file_size: u64,

    // The last wal id.
    next_wal_id: u32,

    // The last flushed sequence number.
    last_flushed_seq_no: i64,

    // The last synced sequence number.
    last_synced_seq_no: AtomicI64,

    synced_seq_no_rx: watch::Receiver<i64>,
    synced_seq_no_tx: watch::Sender<i64>,
}

impl Wal {
    pub async fn open(
        env: Arc<Box<dyn Env>>,
        directory: &str,
        max_file_size: u64,
        last_flushed_seq_no: i64,
    ) -> Result<Self> {
        if !std::path::Path::new(directory).exists() {
            info!("Wal directory {} does not exist, creating it", directory);
            std::fs::create_dir_all(directory)?;
        }

        let mut files = VecDeque::new();

        // Get all files in the directory. Each file will have the name of wal.<file_id>
        let mut file_paths = std::fs::read_dir(directory)?
            .map(|file_path| file_path.unwrap().path().to_str().unwrap().to_owned())
            .collect::<Vec<_>>();

        // Sort the files by the file id. Be careful that the file id is the last part of the file name.
        file_paths.sort_by_key(|path| path.split(".").last().unwrap().parse::<u32>().unwrap());

        #[allow(clippy::needless_late_init)]
        let next_wal_id;
        if file_paths.is_empty() {
            info!("No wal files found, creating a new one");

            // Create a new wal file
            let file_path = format!("{}/wal.0", directory);
            let wal = WalFile::create(env.clone(), &file_path, last_flushed_seq_no).await?;
            files.push_back(wal);
            next_wal_id = 1;
        } else {
            for file_path in file_paths {
                files.push_back(WalFile::open(env.clone(), &file_path).await?);
            }
            next_wal_id = files.back().unwrap().get_wal_id() + 1;
        }

        let last_synced_seq_no = AtomicI64::new(last_flushed_seq_no);
        let (synced_seq_no_tx, synced_seq_no_rx) =
            watch::channel(last_synced_seq_no.load(Ordering::SeqCst));

        Ok(Self {
            directory: directory.to_string(),
            files,
            env,
            max_file_size,
            next_wal_id,
            last_flushed_seq_no,
            last_synced_seq_no,
            synced_seq_no_rx,
            synced_seq_no_tx,
        })
    }

    pub fn get_iterators(&self) -> Vec<WalFileIterator> {
        self.files
            .iter()
            .map(|file| file.get_iterator().unwrap())
            .collect()
    }

    pub async fn append(
        &mut self,
        doc_ids: &[u128],
        user_ids: &[u128],
        op_type: WalOpType<&[f32]>,
        attributes: Option<Arc<Vec<DocumentAttribute>>>,
    ) -> Result<u64> {
        let last_file = self.files.back().unwrap();
        if last_file.get_file_size()? >= self.max_file_size {
            let seq_no = last_file.get_last_seq_no();
            let file_path = format!("{}/wal.{}", self.directory, self.next_wal_id);
            let wal = WalFile::create(self.env.clone(), &file_path, seq_no).await?;
            self.files.push_back(wal);
            self.next_wal_id += 1;
        }

        match self
            .files
            .back_mut()
            .unwrap()
            .append(doc_ids, user_ids, op_type, attributes)
            .await
        {
            Ok(seq_no) => {
                self.last_flushed_seq_no = seq_no.try_into().unwrap();
                Ok(seq_no)
            }
            Err(e) => Err(e),
        }
    }

    /// Append a new entry to the wal. If the last file is full, create a new file.
    pub async fn append_raw(&mut self, data: &[u8]) -> Result<u64> {
        let last_file = self.files.back().unwrap();
        if last_file.get_file_size()? >= self.max_file_size {
            let seq_no = last_file.get_last_seq_no();
            let file_path = format!("{}/wal.{}", self.directory, self.next_wal_id);
            let wal = WalFile::create(self.env.clone(), &file_path, seq_no).await?;
            self.files.push_back(wal);
            self.next_wal_id += 1;
        }

        match self.files.back_mut().unwrap().append_raw(data).await {
            Ok(seq_no) => {
                self.last_flushed_seq_no = seq_no.try_into().unwrap();
                Ok(seq_no)
            }
            Err(e) => Err(e),
        }
    }

    pub fn trim_wal(&mut self, flushed_seq_no: i64) -> Result<()> {
        let mut idx = 0;
        for file in self.files.iter() {
            if file.get_last_seq_no() > flushed_seq_no {
                break;
            }
            idx += 1;
        }
        for _ in 0..idx {
            // Don't trim everything.
            if self.files.len() == 1 {
                break;
            }
            let wal = self.files.pop_front().unwrap();
            std::fs::remove_file(wal.get_path())?;
        }
        Ok(())
    }

    pub fn get_rx(&self) -> watch::Receiver<i64> {
        self.synced_seq_no_rx.clone()
    }

    pub async fn sync(&self) -> Result<u64> {
        let flushed_seq_no = self.last_flushed_seq_no;
        let current_synced_seq_no = self.last_synced_seq_no.load(Ordering::SeqCst);

        // Return early if there is no need to sync
        if flushed_seq_no == current_synced_seq_no {
            return Ok(0);
        }

        // Sync files that contain flushed_seq_no or have sequence numbers less than flushed_seq_no
        for file in self.files.iter() {
            // let file_start_seq_no = file.get_start_seq_no();
            let file_last_seq_no = file.get_last_seq_no();

            // // Check if flushed_seq_no is within the range of the file or less than the file's range
            // if file_start_seq_no <= current_synced_seq_no
            //     && file_last_seq_no >= current_synced_seq_no
            // {
            //     file.sync_data().await?;
            // } else if file_start_seq_no > current_synced_seq_no {
            //     file.sync_data().await?;
            // }

            // Check if file needs to be synced based on current_synced_seq_no
            if file_last_seq_no >= current_synced_seq_no {
                file.sync_data().await?;
            }
        }

        self.last_synced_seq_no
            .store(flushed_seq_no, Ordering::SeqCst);
        self.synced_seq_no_tx.send(flushed_seq_no).unwrap();

        // The number of entries synced is the difference between flushed and current synced sequence numbers
        Ok((flushed_seq_no - current_synced_seq_no) as u64)
    }

    pub fn get_synced_seq_no(&self) -> watch::Receiver<i64> {
        self.synced_seq_no_rx.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_env() -> Arc<Box<dyn Env>> {
        let env_config = utils::file_io::env::EnvConfig::default();
        Arc::new(Box::new(utils::file_io::env::DefaultEnv::new(env_config)))
    }

    #[tokio::test]
    async fn test_wal() {
        // Create 3 wal files. First one has seq_no from 0 to 9, second one has seq_no from 10 to 19, and the third one has seq_no from 20 to 29.
        let tmp_dir = tempdir::TempDir::new("wal_test").unwrap();
        let dir = tmp_dir.path();
        let env = create_env();
        for i in 0..3 {
            let file_path = dir.join(format!("wal.{}", i));
            let mut wal = WalFile::create(env.clone(), file_path.to_str().unwrap(), i * 10 - 1)
                .await
                .unwrap();
            for j in 0..10 {
                wal.append_raw(format!("hello_{}", j).as_bytes())
                    .await
                    .unwrap();
            }
        }

        let wal = Wal::open(env, dir.to_str().unwrap(), 1024, -1)
            .await
            .unwrap();
        let iterators = wal.get_iterators();
        for (i, iterator) in iterators.iter().enumerate() {
            assert_eq!(iterator.last_seq_no(), (i as i64 + 1) * 10 - 1);
        }
    }

    #[tokio::test]
    async fn test_wal_open_empty_dir() {
        let tmp_dir = tempdir::TempDir::new("wal_test").unwrap();
        let dir = tmp_dir.path();
        let env = create_env();
        let wal = Wal::open(env, dir.to_str().unwrap(), 1024, -1)
            .await
            .unwrap();
        assert_eq!(wal.files.len(), 1);
        assert_eq!(wal.files[0].get_num_entries(), 0);
    }

    #[tokio::test]
    async fn test_wal_append_raw() {
        let tmp_dir = tempdir::TempDir::new("wal_test").unwrap();
        let dir = tmp_dir.path();
        let env = create_env();
        let mut wal = Wal::open(env, dir.to_str().unwrap(), 1024, -1)
            .await
            .unwrap();
        for i in 0..100 {
            let seq_no = wal
                .append_raw(format!("hello{}", i).as_bytes())
                .await
                .unwrap();
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

    #[tokio::test]
    async fn test_wal_append() {
        let tmp_dir = tempdir::TempDir::new("wal_test").unwrap();
        let dir = tmp_dir.path();
        let env = create_env();
        let mut wal = Wal::open(env, dir.to_str().unwrap(), 1024, -1)
            .await
            .unwrap();
        for i in 0..5 {
            let data = vec![i as f32; 10];
            let seq_no = wal
                .append(&[i as u128], &[i as u128], WalOpType::Insert(&data), None)
                .await
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

            match decoded.op_type {
                WalOpType::Insert(data) => {
                    assert_eq!(data, vec![i as f32; 10]);
                }
                WalOpType::Delete => panic!("Expected Insert variant"),
            }
        }
    }

    #[tokio::test]
    async fn test_wal_append_delete() {
        let tmp_dir = tempdir::TempDir::new("wal_test").unwrap();
        let dir = tmp_dir.path();
        let env = create_env();
        let mut wal = Wal::open(env, dir.to_str().unwrap(), 1024, -1)
            .await
            .unwrap();
        for i in 0..5 {
            let seq_no = wal
                .append(&[i as u128], &[i as u128], WalOpType::Delete, None)
                .await
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
            match decoded.op_type {
                WalOpType::Insert(_) => panic!("Expected Delete variant"),
                WalOpType::Delete => {}
            }
        }
    }

    #[tokio::test]
    async fn test_wal_trim() {
        let tmp_dir = tempdir::TempDir::new("wal_test").unwrap();
        let dir = tmp_dir.path();
        let env = create_env();
        let mut wal = Wal::open(env, dir.to_str().unwrap(), 20, -1).await.unwrap();
        for i in 0..10 {
            wal.append_raw(format!("hello{}", i).as_bytes())
                .await
                .unwrap();
        }

        assert_eq!(wal.files.len(), 10);

        wal.trim_wal(5).unwrap();
        assert_eq!(wal.files.len(), 4);

        let iterators = wal.get_iterators();
        let mut read_seq_nos = vec![];
        for it in iterators {
            for i in it {
                read_seq_nos.push(i.unwrap().seq_no);
            }
        }
        assert_eq!(read_seq_nos, vec![6, 7, 8, 9]);
    }

    #[tokio::test]
    async fn test_wal_trim_empty_dir() {
        let tmp_dir = tempdir::TempDir::new("wal_test").unwrap();
        let dir = tmp_dir.path();
        let env = create_env();
        let mut wal = Wal::open(env, dir.to_str().unwrap(), 1024, -1)
            .await
            .unwrap();
        wal.trim_wal(-1).unwrap();
        assert_eq!(wal.files.len(), 1);

        let file_paths = std::fs::read_dir(dir).unwrap();
        assert_eq!(file_paths.count(), 1);
    }

    #[tokio::test]
    async fn test_wal_open_with_last_flushed_seq_no() {
        let tmp_dir = tempdir::TempDir::new("wal_test").unwrap();
        let dir = tmp_dir.path();
        let env = create_env();
        {
            let mut wal = Wal::open(env.clone(), dir.to_str().unwrap(), 50, 10)
                .await
                .unwrap();
            assert_eq!(wal.last_flushed_seq_no, 10);

            // append 20 entries
            for i in 0..20 {
                wal.append_raw(format!("hello{}", i).as_bytes())
                    .await
                    .unwrap();
            }
            assert_eq!(wal.files.len(), 5);
            assert_eq!(wal.files[0].get_num_entries(), 4);
            assert_eq!(wal.files[0].get_last_seq_no(), 14);

            // now, trim up to seq_no 15
            wal.trim_wal(15).unwrap();
            assert_eq!(wal.files.len(), 4);
            assert_eq!(wal.files.iter().last().unwrap().get_wal_id(), 4);

            // append 20 more entries
            for i in 0..20 {
                wal.append_raw(format!("hello{}", i).as_bytes())
                    .await
                    .unwrap();
            }
            assert_eq!(wal.files.len(), 9);
            assert_eq!(wal.files.iter().last().unwrap().get_wal_id(), 9);
            assert_eq!(wal.files.iter().last().unwrap().get_last_seq_no(), 50);

            // now, trim up to seq_no 50
            wal.trim_wal(50).unwrap();
            assert_eq!(wal.files.len(), 1);
            assert_eq!(wal.files.iter().last().unwrap().get_wal_id(), 9);
            assert_eq!(wal.files.iter().last().unwrap().get_last_seq_no(), 50);
        }

        {
            let wal = Wal::open(env, dir.to_str().unwrap(), 50, 50).await.unwrap();
            assert_eq!(wal.last_flushed_seq_no, 50);
            assert_eq!(wal.files.len(), 1);
            assert_eq!(wal.files.iter().last().unwrap().get_wal_id(), 9);
            assert_eq!(wal.files.iter().last().unwrap().get_last_seq_no(), 50);
        }
    }
}
