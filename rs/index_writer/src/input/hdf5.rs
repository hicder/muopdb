use std::cmp::min;

use anyhow::Result;
use log::error;
use ndarray::s;

use super::{Input, Row};

pub struct Hdf5Reader {
    dataset: hdf5::Dataset,
    chunk_size: usize,
    row_idx: usize,
    num_rows: usize,
    chunk: Vec<Vec<f32>>,

    first_chunk_fetched: bool,
}

impl Hdf5Reader {
    pub fn new(chunk_size: usize, dataset: &str, file: &str) -> Result<Self> {
        let file = hdf5::File::open(file)?;
        let dataset = file.dataset(dataset)?;
        let num_rows = dataset.shape()[0];
        Ok(Self {
            dataset,
            chunk_size,
            row_idx: 0,
            num_rows,
            chunk: vec![],
            first_chunk_fetched: false,
        })
    }

    pub fn fetch_next_chunk(&mut self) {
        self.first_chunk_fetched = true;
        self.chunk.clear();
        let end_idx = min(self.row_idx + self.chunk_size, self.num_rows);
        let selection = s![self.row_idx..end_idx, ..];
        match self.dataset.read_slice_2d(selection) {
            Ok(chunk) => {
                self.chunk = chunk
                    .axis_iter(ndarray::Axis(0))
                    .map(|row| row.to_vec())
                    .collect();
            }
            Err(e) => {
                error!("Failed to read slice from dataset: {}", e);
            }
        }
    }
}

impl Input for Hdf5Reader {
    fn reset(&mut self) {
        self.row_idx = 0;
        self.chunk.clear();
    }

    fn has_next(&self) -> bool {
        self.row_idx < self.num_rows
    }

    // Caller is responsible for ensuring that the chunk is not empty
    fn next(&mut self) -> Row<'_> {
        if self.row_idx % self.chunk_size == 0 {
            self.fetch_next_chunk();
        }

        let idx = self.row_idx % self.chunk_size;
        let doc_id = self.row_idx as u64;
        let slice = &self.chunk[idx];
        self.row_idx += 1;
        Row {
            id: doc_id,
            data: slice,
        }
    }

    fn num_rows(&self) -> usize {
        self.num_rows
    }

    fn skip_to(&mut self, row_idx: usize) {
        let current_chunk_idx = self.row_idx / self.chunk_size;
        let new_chunk_idx = row_idx / self.chunk_size;
        if new_chunk_idx != current_chunk_idx || !self.first_chunk_fetched {
            self.row_idx = row_idx / self.chunk_size * self.chunk_size;
            self.fetch_next_chunk();
        }

        self.row_idx = row_idx;
    }
}

// test
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hdf5_reader() {
        env_logger::init();
        let path = format!("{}/resources/test.hdf5", env!("CARGO_MANIFEST_DIR"));
        let mut reader = Hdf5Reader::new(101, "test", &path).expect("Failed to create Hdf5Reader");
        let mut it = 0;
        while reader.has_next() {
            let _ = reader.next();
            it += 1;
        }

        assert_eq!(it, 1000);
    }
}
