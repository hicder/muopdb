use anyhow::Result;
use proto::muopdb::{AndFilter, DocumentFilter, IdsFilter, OrFilter};

use crate::query::iters::and_iter::AndIter;
use crate::query::iters::ids_iter::IdsIter;
use crate::query::iters::or_iter::OrIter;
use crate::query::iters::term_iter::TermIter;
use crate::query::iters::Iter;
use crate::terms::index::TermIndex;

#[allow(unused)]
pub struct Planner {
    user_id: u128,
    query: DocumentFilter,
    term_index: TermIndex,
}

impl Planner {
    pub fn new(user_id: u128, query: DocumentFilter, term_directory: &str) -> Result<Self> {
        let index_file_path = format!("{}/user_{}/combined", term_directory, user_id);
        if !std::path::Path::new(&index_file_path).exists() {
            return Err(anyhow::anyhow!(
                "Term index file does not exist for user {}: {}",
                user_id,
                index_file_path
            ));
        }

        let read_len = std::fs::metadata(&index_file_path)?.len() as usize;
        let term_index = TermIndex::new(index_file_path, 0, read_len)?;
        Ok(Self {
            user_id,
            query,
            term_index,
        })
    }

    pub fn plan(&self) -> Result<Iter> {
        self.plan_filter(&self.query)
    }

    fn plan_filter(&self, filter: &DocumentFilter) -> Result<Iter> {
        use proto::muopdb::document_filter::Filter;

        match filter.filter.as_ref() {
            Some(Filter::And(and_filter)) => self.plan_and_filter(and_filter),
            Some(Filter::Or(or_filter)) => self.plan_or_filter(or_filter),
            Some(Filter::Ids(ids_filter)) => self.plan_ids_filter(ids_filter),
            Some(Filter::Contains(contains_filter)) => {
                let term = format!("{}:{}", contains_filter.path, contains_filter.value);
                if let Some(term_id) = self.term_index.get_term_id(&term) {
                    let posting_list = self.term_index.get_posting_list_iterator(term_id)?;
                    // TODO(hung): This posting list contains point IDs for the term.
                    // We need to find a way to convert these point IDs to document IDs.
                    // For now, we return the posting list directly.
                    Ok(Iter::Term(TermIter::new(posting_list)))
                } else {
                    // Term not found - return an iterator that yields no results
                    Ok(Iter::Ids(IdsIter::new(vec![])))
                }
            }
            Some(Filter::NotContains(_not_contains_filter)) => {
                // Skip NotContains filter for now as requested
                todo!("NotContainsFilter not yet implemented")
            }
            None => {
                // Empty filter - return an iterator that yields no results
                Ok(Iter::Ids(IdsIter::new(vec![])))
            }
        }
    }

    fn plan_and_filter(&self, and_filter: &AndFilter) -> Result<Iter> {
        if and_filter.filters.is_empty() {
            // Empty AND filter - return an iterator that yields no results
            return Ok(Iter::Ids(IdsIter::new(vec![])));
        }

        let mut iters = Vec::new();
        for filter in &and_filter.filters {
            iters.push(self.plan_filter(filter)?);
        }

        Ok(Iter::And(AndIter::new(iters)))
    }

    fn plan_or_filter(&self, or_filter: &OrFilter) -> Result<Iter> {
        if or_filter.filters.is_empty() {
            // Empty OR filter - return an iterator that yields no results
            return Ok(Iter::Ids(IdsIter::new(vec![])));
        }

        let mut iters = Vec::new();
        for filter in &or_filter.filters {
            iters.push(self.plan_filter(filter)?);
        }

        Ok(Iter::Or(OrIter::new(iters)))
    }

    fn plan_ids_filter(&self, ids_filter: &IdsFilter) -> Result<Iter> {
        let mut ids = Vec::new();
        for id in &ids_filter.ids {
            // Convert the 128-bit ID (split into two 64-bit parts) to a single u128
            let full_id = ((id.high_id as u128) << 64) | (id.low_id as u128);
            ids.push(full_id);
        }

        // Sort the IDs to maintain the expected order for the IdsIter
        ids.sort_unstable();

        Ok(Iter::Ids(IdsIter::new(ids)))
    }
}
#[cfg(test)]
mod tests {
    use std::fs;

    use proto::muopdb::{AndFilter, Id, IdsFilter, OrFilter};

    use super::*;
    use crate::query::iters::InvertedIndexIter;
    use crate::terms::builder::MultiTermBuilder;
    use crate::terms::writer::MultiTermWriter;

    fn create_test_term_index(temp_dir: &tempdir::TempDir) -> (String, u128) {
        let base_dir = temp_dir.path().to_str().unwrap();
        let user_id = 12345u128;
        let term_dir = format!("{}/terms", base_dir);

        // Create directories
        fs::create_dir_all(&term_dir).unwrap();

        let mut multi_builder =
            MultiTermBuilder::new(temp_dir.path().to_str().unwrap().to_string());
        multi_builder
            .add(user_id, 1, "test:term".to_string())
            .unwrap();
        multi_builder.build().unwrap();

        let multi_writer = MultiTermWriter::new(term_dir.clone());
        multi_writer.write(&mut multi_builder).unwrap();
        assert!(fs::metadata(format!("{}/user_{}/combined", term_dir, user_id)).is_ok());
        (term_dir, user_id)
    }

    #[test]
    fn test_plan_ids_filter() {
        let temp_dir = tempdir::TempDir::new("term_index_test").unwrap();
        let (term_dir, user_id) = create_test_term_index(&temp_dir);

        let ids_filter = IdsFilter {
            ids: vec![
                Id {
                    low_id: 1,
                    high_id: 0,
                },
                Id {
                    low_id: 3,
                    high_id: 0,
                },
                Id {
                    low_id: 5,
                    high_id: 0,
                },
            ],
        };

        let document_filter = DocumentFilter {
            filter: Some(proto::muopdb::document_filter::Filter::Ids(ids_filter)),
        };

        let planner = Planner::new(user_id, document_filter, &term_dir).unwrap();
        let result = planner.plan();

        assert!(result.is_ok());
        let mut iter = result.unwrap();

        // Should return the IDs in sorted order
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next(), Some(3));
        assert_eq!(iter.next(), Some(5));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_plan_and_filter() {
        let temp_dir = tempdir::TempDir::new("term_index_test").unwrap();
        let (term_dir, user_id) = create_test_term_index(&temp_dir);

        let ids_filter1 = IdsFilter {
            ids: vec![Id {
                low_id: 1,
                high_id: 0,
            }],
        };

        let ids_filter2 = IdsFilter {
            ids: vec![Id {
                low_id: 2,
                high_id: 0,
            }],
        };

        let and_filter = AndFilter {
            filters: vec![
                DocumentFilter {
                    filter: Some(proto::muopdb::document_filter::Filter::Ids(ids_filter1)),
                },
                DocumentFilter {
                    filter: Some(proto::muopdb::document_filter::Filter::Ids(ids_filter2)),
                },
            ],
        };

        let document_filter = DocumentFilter {
            filter: Some(proto::muopdb::document_filter::Filter::And(and_filter)),
        };

        let planner = Planner::new(user_id, document_filter, &term_dir).unwrap();
        let result = planner.plan();

        assert!(result.is_ok());
        // The AndIter should be created successfully (even though its methods are todo!())
        let _iter = result.unwrap();
    }

    #[test]
    fn test_plan_or_filter() {
        let temp_dir = tempdir::TempDir::new("term_index_test").unwrap();
        let (term_dir, user_id) = create_test_term_index(&temp_dir);

        let ids_filter1 = IdsFilter {
            ids: vec![Id {
                low_id: 1,
                high_id: 0,
            }],
        };

        let ids_filter2 = IdsFilter {
            ids: vec![Id {
                low_id: 2,
                high_id: 0,
            }],
        };

        let or_filter = OrFilter {
            filters: vec![
                DocumentFilter {
                    filter: Some(proto::muopdb::document_filter::Filter::Ids(ids_filter1)),
                },
                DocumentFilter {
                    filter: Some(proto::muopdb::document_filter::Filter::Ids(ids_filter2)),
                },
            ],
        };

        let document_filter = DocumentFilter {
            filter: Some(proto::muopdb::document_filter::Filter::Or(or_filter)),
        };

        let planner = Planner::new(user_id, document_filter, &term_dir).unwrap();
        let result = planner.plan();

        assert!(result.is_ok());
        // The OrIter should be created successfully (even though its methods are todo!())
        let _iter = result.unwrap();
    }

    #[test]
    fn test_plan_empty_filter() {
        let temp_dir = tempdir::TempDir::new("term_index_test").unwrap();
        let (term_dir, user_id) = create_test_term_index(&temp_dir);

        let document_filter = DocumentFilter { filter: None };

        let planner = Planner::new(user_id, document_filter, &term_dir).unwrap();
        let result = planner.plan();

        assert!(result.is_ok());
        let mut iter = result.unwrap();

        // Should return no results
        assert_eq!(iter.next(), None);
    }
}
