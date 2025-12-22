use anyhow::Result;
use compression::elias_fano::ef::EliasFanoDecodingIterator;

use crate::query::iters::{InvertedIndexIter, IterState};
use crate::terms::index::TermIndex;

pub struct TermIter<'a> {
    iter: EliasFanoDecodingIterator<'a, u32>,
    state: IterState<u32>,
}

impl<'a> TermIter<'a> {
    pub fn new(term_index: &'a TermIndex, term_id: u64) -> Result<Self> {
        let iter = term_index.get_posting_list_iterator(term_id)?;
        Ok(Self {
            iter,
            state: IterState::NotStarted,
        })
    }
}

impl<'a> InvertedIndexIter for TermIter<'a> {
    fn next(&mut self) -> Option<u32> {
        match self.state {
            IterState::NotStarted => {
                let point = self.iter.current();
                self.iter.next();
                self.state = if point.is_some() {
                    IterState::At(point.unwrap())
                } else {
                    IterState::Exhausted
                };
                point
            }
            IterState::At(_) => {
                let point = self.iter.current();
                self.iter.next();
                self.state = if point.is_some() {
                    IterState::At(point.unwrap())
                } else {
                    IterState::Exhausted
                };
                point
            }
            IterState::Exhausted => None,
        }
    }

    fn skip_to(&mut self, point_id: u32) {
        self.iter.skip_to(point_id);
        let point = self.iter.current();
        self.state = if point.is_some() {
            IterState::At(point.unwrap())
        } else {
            IterState::Exhausted
        };
    }

    fn point_id(&mut self) -> Option<u32> {
        match self.state {
            IterState::At(point) => Some(point),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terms::builder::TermBuilder;
    use crate::terms::writer::TermWriter;

    #[test]
    fn test_term_iter_next() {
        let temp_dir = tempdir::TempDir::new("test_term_iter").unwrap();
        let base_directory = temp_dir.path();
        let base_dir_str = base_directory.to_str().unwrap().to_string();

        let mut builder = TermBuilder::new().unwrap();
        builder.add(0, "apple".to_string()).unwrap();
        builder.add(1, "apple".to_string()).unwrap();
        builder.add(2, "apple".to_string()).unwrap();
        builder.add(3, "apple".to_string()).unwrap();
        builder.add(5, "apple".to_string()).unwrap();
        builder.add(0, "banana".to_string()).unwrap();
        builder.add(2, "banana".to_string()).unwrap();

        builder.build().unwrap();
        let writer = TermWriter::new(base_dir_str.clone());
        writer.write(&mut builder).unwrap();

        let path = format!("{base_dir_str}/combined");
        let file_len = std::fs::metadata(&path).unwrap().len();
        let index = TermIndex::new(path, 0, file_len as usize).unwrap();

        let apple_id = index.get_term_id("apple").unwrap();
        let mut iter = TermIter::new(&index, apple_id).unwrap();

        let val1 = iter.next();
        let val2 = iter.next();
        let val3 = iter.next();

        assert_eq!(val1, Some(0));
        assert_eq!(val2, Some(1));
        assert_eq!(val3, Some(2));
    }

    #[test]
    fn test_term_iter_skip_to() {
        let temp_dir = tempdir::TempDir::new("test_term_iter_skip").unwrap();
        let base_directory = temp_dir.path();
        let base_dir_str = base_directory.to_str().unwrap().to_string();

        let mut builder = TermBuilder::new().unwrap();
        for i in [0, 2, 5, 7, 10, 15, 20] {
            builder.add(i, "apple".to_string()).unwrap();
        }

        builder.build().unwrap();
        let writer = TermWriter::new(base_dir_str.clone());
        writer.write(&mut builder).unwrap();

        let path = format!("{base_dir_str}/combined");
        let file_len = std::fs::metadata(&path).unwrap().len();
        let index = TermIndex::new(path, 0, file_len as usize).unwrap();

        let apple_id = index.get_term_id("apple").unwrap();
        let mut iter = TermIter::new(&index, apple_id).unwrap();

        iter.skip_to(5);
        assert_eq!(iter.next(), Some(5));
        assert_eq!(iter.next(), Some(7));

        iter.skip_to(12);
        assert_eq!(iter.point_id(), Some(15));
        assert_eq!(iter.next(), Some(15));

        iter.skip_to(30);
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_term_iter_skip_to_basic() {
        let temp_dir = tempdir::TempDir::new("test_term_iter_skip").unwrap();
        let base_directory = temp_dir.path();
        let base_dir_str = base_directory.to_str().unwrap().to_string();

        let mut builder = TermBuilder::new().unwrap();
        for i in [0] {
            builder.add(i, "apple".to_string()).unwrap();
        }

        builder.build().unwrap();
        let writer = TermWriter::new(base_dir_str.clone());
        writer.write(&mut builder).unwrap();

        let path = format!("{base_dir_str}/combined");
        let file_len = std::fs::metadata(&path).unwrap().len();
        let index = TermIndex::new(path, 0, file_len as usize).unwrap();

        let apple_id = index.get_term_id("apple").unwrap();
        let mut iter = TermIter::new(&index, apple_id).unwrap();

        iter.next();
        assert_eq!(iter.point_id(), Some(0));
    }

    #[test]
    fn test_term_iter_point_id() {
        let temp_dir = tempdir::TempDir::new("test_term_iter_point_id").unwrap();
        let base_directory = temp_dir.path();
        let base_dir_str = base_directory.to_str().unwrap().to_string();

        let mut builder = TermBuilder::new().unwrap();
        builder.add(1, "apple".to_string()).unwrap();
        builder.add(3, "apple".to_string()).unwrap();
        builder.add(5, "apple".to_string()).unwrap();

        builder.build().unwrap();
        let writer = TermWriter::new(base_dir_str.clone());
        writer.write(&mut builder).unwrap();

        let path = format!("{base_dir_str}/combined");
        let file_len = std::fs::metadata(&path).unwrap().len();
        let index = TermIndex::new(path, 0, file_len as usize).unwrap();

        let apple_id = index.get_term_id("apple").unwrap();
        let mut iter = TermIter::new(&index, apple_id).unwrap();

        assert_eq!(iter.point_id(), None);
        iter.next();
        assert_eq!(iter.point_id(), Some(1));
        iter.next();
        assert_eq!(iter.point_id(), Some(3));
        iter.next();
        assert_eq!(iter.point_id(), Some(5));
        iter.next();
        assert_eq!(iter.point_id(), None);
    }

    #[test]
    fn test_term_iter_combined() {
        let temp_dir = tempdir::TempDir::new("test_term_iter_combined").unwrap();
        let base_directory = temp_dir.path();
        let base_dir_str = base_directory.to_str().unwrap().to_string();

        let mut builder = TermBuilder::new().unwrap();
        for i in [0, 2, 4, 6, 8, 10] {
            builder.add(i, "apple".to_string()).unwrap();
        }

        builder.build().unwrap();
        let writer = TermWriter::new(base_dir_str.clone());
        writer.write(&mut builder).unwrap();

        let path = format!("{base_dir_str}/combined");
        let file_len = std::fs::metadata(&path).unwrap().len();
        let index = TermIndex::new(path, 0, file_len as usize).unwrap();

        let apple_id = index.get_term_id("apple").unwrap();
        let mut iter = TermIter::new(&index, apple_id).unwrap();

        assert_eq!(iter.next(), Some(0));
        assert_eq!(iter.point_id(), Some(0));

        iter.skip_to(5);
        assert_eq!(iter.next(), Some(6));
        assert_eq!(iter.point_id(), Some(6));

        assert_eq!(iter.next(), Some(8));
        assert_eq!(iter.point_id(), Some(8));

        iter.skip_to(20);
        assert_eq!(iter.next(), None);
        assert_eq!(iter.point_id(), None);
    }

    #[test]
    fn test_term_iter_empty_posting_list() {
        let temp_dir = tempdir::TempDir::new("test_term_iter_empty").unwrap();
        let base_directory = temp_dir.path();
        let base_dir_str = base_directory.to_str().unwrap().to_string();

        let mut builder = TermBuilder::new().unwrap();
        builder.add(0, "apple".to_string()).unwrap();

        builder.build().unwrap();
        let writer = TermWriter::new(base_dir_str.clone());
        writer.write(&mut builder).unwrap();

        let path = format!("{base_dir_str}/combined");
        let file_len = std::fs::metadata(&path).unwrap().len();
        let index = TermIndex::new(path, 0, file_len as usize).unwrap();

        let apple_id = index.get_term_id("apple").unwrap();
        let mut iter = TermIter::new(&index, apple_id).unwrap();

        assert_eq!(iter.next(), Some(0));
        assert_eq!(iter.next(), None);

        iter.skip_to(10);
        assert_eq!(iter.next(), None);
    }
}
