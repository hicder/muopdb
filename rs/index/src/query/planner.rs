use anyhow::Result;
use proto::muopdb::{AndFilter, DocumentFilter, IdsFilter, OrFilter};

use crate::query::iter::Iter;
use crate::query::iters::and_iter::AndIter;
use crate::query::iters::ids_iter::IdsIter;
use crate::query::iters::or_iter::OrIter;

#[allow(unused)]
pub struct Planner {
    query: DocumentFilter,
}

impl Planner {
    pub fn new(query: DocumentFilter) -> Self {
        Self { query }
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
            Some(Filter::Contains(_contains_filter)) => {
                // TODO: Implement ContainsFilter handling
                // This would require access to the terms index to:
                // 1. Look up the term ID for the given path and value
                // 2. Get the posting list iterator for that term
                // 3. Create a TermIter from the posting list
                todo!("ContainsFilter not yet implemented")
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
    use proto::muopdb::{AndFilter, Id, IdsFilter, OrFilter};

    use super::*;
    use crate::query::iter::InvertedIndexIter;

    #[test]
    fn test_plan_ids_filter() {
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

        let planner = Planner::new(document_filter);
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

        let planner = Planner::new(document_filter);
        let result = planner.plan();

        assert!(result.is_ok());
        // The AndIter should be created successfully (even though its methods are todo!())
        let _iter = result.unwrap();
    }

    #[test]
    fn test_plan_or_filter() {
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

        let planner = Planner::new(document_filter);
        let result = planner.plan();

        assert!(result.is_ok());
        // The OrIter should be created successfully (even though its methods are todo!())
        let _iter = result.unwrap();
    }

    #[test]
    fn test_plan_empty_filter() {
        let document_filter = DocumentFilter { filter: None };

        let planner = Planner::new(document_filter);
        let result = planner.plan();

        assert!(result.is_ok());
        let mut iter = result.unwrap();

        // Should return no results
        assert_eq!(iter.next(), None);
    }
}
