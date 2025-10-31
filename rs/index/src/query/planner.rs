use std::sync::Arc;

use anyhow::Result;
use proto::muopdb::{AndFilter, DocumentFilter, IdsFilter, OrFilter};

use crate::multi_terms::index::MultiTermIndex;
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
    term_index: Arc<TermIndex>,
}

impl Planner {
    pub fn new(
        user_id: u128,
        query: DocumentFilter,
        multi_term_index: Arc<MultiTermIndex>,
    ) -> Result<Self> {
        let term_index = multi_term_index.get_or_create_index(user_id)?;
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
                    Ok(Iter::Term(TermIter::new(&self.term_index, term_id)?))
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
            ids.push(*id);
        }

        // Sort the IDs to maintain the expected order for the IdsIter
        ids.sort_unstable();

        Ok(Iter::Ids(IdsIter::new(ids)))
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use config::collection::CollectionConfig;
    use proto::muopdb::{AndFilter, IdsFilter, OrFilter};
    use utils::test_utils::generate_random_vector;

    use super::*;
    use crate::multi_spann::builder::MultiSpannBuilder;
    use crate::multi_terms::builder::MultiTermBuilder;
    use crate::multi_terms::writer::MultiTermWriter;
    use crate::query::iters::InvertedIndexIter;

    fn create_test_spann_index(
        base_directory: String,
        user_id: u128,
        doc_ids: &[u128],
    ) -> Vec<u32> {
        let num_features = 4;
        let collection_config = CollectionConfig {
            num_features,
            ..CollectionConfig::default_test_config()
        };
        let multi_spann_builder =
            MultiSpannBuilder::new(collection_config, base_directory.clone()).unwrap();

        doc_ids
            .iter()
            .map(|&doc_id| {
                multi_spann_builder
                    .insert(user_id, doc_id, &generate_random_vector(num_features))
                    .unwrap()
            })
            .collect::<Vec<u32>>()
    }

    fn create_test_term_index(
        base_directory: String,
        user_id: u128,
        point_ids: &[u32],
    ) -> MultiTermIndex {
        // Create a MultiTermIndex with some test data
        let multi_builder = MultiTermBuilder::new();
        // Insert terms for the user with the provided point IDs
        point_ids.iter().for_each(|&pid| {
            multi_builder
                .add(user_id, pid, format!("field:term{}", pid))
                .unwrap();
        });
        multi_builder.build().unwrap();

        let multi_writer = MultiTermWriter::new(base_directory.clone());
        multi_writer.write(&multi_builder).unwrap();

        MultiTermIndex::new(base_directory).unwrap()
    }

    fn create_test_indexes(temp_dir: &tempdir::TempDir) -> (Arc<MultiTermIndex>, u128) {
        let base_dir = temp_dir.path().to_str().unwrap();
        let term_dir = format!("{}/terms", base_dir);

        // Create term directory
        fs::create_dir_all(&term_dir).unwrap();

        let user_id = 12345u128;
        let doc_ids = [1u128, 2, 3, 4, 5];
        let point_ids = create_test_spann_index(base_dir.to_string(), user_id, &doc_ids);
        let multi_term_index = create_test_term_index(term_dir, user_id, &point_ids);
        (Arc::new(multi_term_index), user_id)
    }

    #[test]
    fn test_plan_ids_filter() {
        let temp_dir = tempdir::TempDir::new("term_index_test").unwrap();
        let (multi_term_index, user_id) = create_test_indexes(&temp_dir);

        let ids_filter = IdsFilter { ids: vec![1, 3, 5] };

        let document_filter = DocumentFilter {
            filter: Some(proto::muopdb::document_filter::Filter::Ids(ids_filter)),
        };

        let planner = Planner::new(user_id, document_filter, multi_term_index).unwrap();
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
        let (multi_term_index, user_id) = create_test_indexes(&temp_dir);

        let ids_filter1 = IdsFilter { ids: vec![1] };

        let ids_filter2 = IdsFilter { ids: vec![2] };

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

        let planner = Planner::new(user_id, document_filter, multi_term_index).unwrap();
        let result = planner.plan();

        assert!(result.is_ok());
        // The AndIter should be created successfully
        let _iter = result.unwrap();
    }

    #[test]
    fn test_plan_or_filter() {
        let temp_dir = tempdir::TempDir::new("term_index_test").unwrap();
        let (multi_term_index, user_id) = create_test_indexes(&temp_dir);

        let ids_filter1 = IdsFilter { ids: vec![1] };

        let ids_filter2 = IdsFilter { ids: vec![2] };

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

        let planner = Planner::new(user_id, document_filter, multi_term_index).unwrap();
        let result = planner.plan();

        assert!(result.is_ok());
        // The OrIter should be created successfully (even though its methods are todo!())
        let _iter = result.unwrap();
    }

    #[test]
    fn test_plan_empty_filter() {
        let temp_dir = tempdir::TempDir::new("term_index_test").unwrap();
        let (multi_term_index, user_id) = create_test_indexes(&temp_dir);

        let document_filter = DocumentFilter { filter: None };

        let planner = Planner::new(user_id, document_filter, multi_term_index).unwrap();
        let result = planner.plan();

        assert!(result.is_ok());
        let mut iter = result.unwrap();

        // Should return no results
        assert_eq!(iter.next(), None);
    }
}
