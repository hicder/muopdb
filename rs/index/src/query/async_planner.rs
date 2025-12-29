use std::sync::Arc;

use anyhow::Result;
use config::attribute_schema::{AttributeSchema, AttributeType, Language};
use futures::future::{BoxFuture, FutureExt};
use proto::muopdb::{AndFilter, DocumentFilter, IdsFilter, OrFilter};

use crate::multi_terms::index::MultiTermIndex;
use crate::query::async_iters::{AsyncAndIter, AsyncIdsIter, AsyncIter, AsyncOrIter, AsyncTermIter};
use crate::terms::index::TermIndex;
use crate::tokenizer::stemming_tokenizer::StemmingTokenizer;
use crate::tokenizer::tokenizer::{TokenStream, Tokenizer};

#[allow(dead_code)]
pub struct AsyncPlanner {
    user_id: u128,
    query: DocumentFilter,
    term_index: Arc<TermIndex>,
    attribute_schema: Option<AttributeSchema>,
}

impl AsyncPlanner {
    pub async fn new(
        user_id: u128,
        query: DocumentFilter,
        multi_term_index: Arc<MultiTermIndex>,
        attribute_schema: Option<AttributeSchema>,
    ) -> Result<Self> {
        let term_index = multi_term_index.get_or_create_index_async(user_id).await?;
        Ok(Self {
            user_id,
            query,
            term_index,
            attribute_schema,
        })
    }

    pub async fn plan(&self) -> Result<AsyncIter> {
        self.plan_filter(&self.query).await
    }

    pub async fn plan_with_ids(&self, extra_ids: &[u32]) -> Result<AsyncIter> {
        let doc_filter_iter = self.plan_filter(&self.query).await?;

        if extra_ids.is_empty() {
            return Ok(doc_filter_iter);
        }

        let mut ids = extra_ids.to_vec();
        ids.sort_unstable();
        ids.dedup();

        let extra_ids_iter = AsyncIter::Ids(AsyncIdsIter::new(ids));
        Ok(AsyncIter::And(AsyncAndIter::new(vec![
            doc_filter_iter,
            extra_ids_iter,
        ])))
    }

    fn plan_filter<'a>(&'a self, filter: &'a DocumentFilter) -> BoxFuture<'a, Result<AsyncIter>> {
        async move {
            use proto::muopdb::document_filter::Filter;

            match filter.filter.as_ref() {
                Some(Filter::And(and_filter)) => self.plan_and_filter(and_filter).await,
                Some(Filter::Or(or_filter)) => self.plan_or_filter(or_filter).await,
                Some(Filter::Ids(ids_filter)) => self.plan_ids_filter(ids_filter).await,
                Some(Filter::Contains(contains_filter)) => {
                    let language = self
                        .attribute_schema
                        .as_ref()
                        .and_then(|s| s.fields.get(&contains_filter.path))
                        .map(|t| match t {
                            AttributeType::Text(l) => *l,
                            _ => Language::English,
                        })
                        .unwrap_or(Language::English);

                    let tokenizer = StemmingTokenizer::for_language(language);
                    let mut stream = tokenizer.input(&contains_filter.value);

                    let mut stemmed_tokens = Vec::new();
                    while let Some(token) = stream.next() {
                        stemmed_tokens.push(token.text);
                    }

                    if stemmed_tokens.is_empty() {
                        return Ok(AsyncIter::Ids(AsyncIdsIter::new(vec![])));
                    }

                    if stemmed_tokens.len() == 1 {
                        let stemmed_value = &stemmed_tokens[0];
                        let term = format!("{}:{}", contains_filter.path, stemmed_value);
                        if let Some(term_id) = self.term_index.get_term_id(&term) {
                            return Ok(AsyncIter::Term(AsyncTermIter::new(
                                self.term_index
                                    .get_posting_list_iterator_block_based(term_id)
                                    .await?,
                            )));
                        } else {
                            return Ok(AsyncIter::Ids(AsyncIdsIter::new(vec![])));
                        }
                    }

                    let mut iters = Vec::new();
                    for stemmed_value in stemmed_tokens {
                        let term = format!("{}:{}", contains_filter.path, stemmed_value);
                        if let Some(term_id) = self.term_index.get_term_id(&term) {
                            iters.push(AsyncIter::Term(AsyncTermIter::new(
                                self.term_index
                                    .get_posting_list_iterator_block_based(term_id)
                                    .await?,
                            )));
                        } else {
                            return Ok(AsyncIter::Ids(AsyncIdsIter::new(vec![])));
                        }
                    }

                    Ok(AsyncIter::And(AsyncAndIter::new(iters)))
                }
                Some(Filter::NotContains(_not_contains_filter)) => {
                    todo!("NotContainsFilter not yet implemented")
                }
                None => Ok(AsyncIter::Ids(AsyncIdsIter::new(vec![]))),
            }
        }
        .boxed()
    }

    async fn plan_and_filter(&self, and_filter: &AndFilter) -> Result<AsyncIter> {
        if and_filter.filters.is_empty() {
            return Ok(AsyncIter::Ids(AsyncIdsIter::new(vec![])));
        }

        let mut iters = Vec::new();
        for filter in &and_filter.filters {
            iters.push(self.plan_filter(filter).await?);
        }

        Ok(AsyncIter::And(AsyncAndIter::new(iters)))
    }

    async fn plan_or_filter(&self, or_filter: &OrFilter) -> Result<AsyncIter> {
        if or_filter.filters.is_empty() {
            return Ok(AsyncIter::Ids(AsyncIdsIter::new(vec![])));
        }

        let mut iters = Vec::new();
        for filter in &or_filter.filters {
            iters.push(self.plan_filter(filter).await?);
        }

        Ok(AsyncIter::Or(AsyncOrIter::new(iters)))
    }

    async fn plan_ids_filter(&self, ids_filter: &IdsFilter) -> Result<AsyncIter> {
        let mut ids = Vec::new();
        for id in &ids_filter.ids {
            ids.push(*id);
        }
        ids.sort_unstable();
        Ok(AsyncIter::Ids(AsyncIdsIter::new(ids)))
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::sync::Arc;
    use tempdir::TempDir;

    use config::collection::CollectionConfig;
    use proto::muopdb::{DocumentFilter, IdsFilter};
    use utils::test_utils::generate_random_vector;

    use crate::multi_spann::builder::MultiSpannBuilder;
    use crate::multi_terms::builder::MultiTermBuilder;
    use crate::multi_terms::index::MultiTermIndex;
    use crate::multi_terms::writer::MultiTermWriter;
    use crate::query::async_iters::AsyncInvertedIndexIter;
    use super::AsyncPlanner;

    async fn create_test_indexes(temp_dir: &TempDir) -> (Arc<MultiTermIndex>, u128, String) {
        let base_dir = temp_dir.path().to_str().unwrap();
        let term_dir = format!("{}/terms", base_dir);
        fs::create_dir_all(&term_dir).unwrap();

        let user_id = 12345u128;
        let num_features = 4;
        let collection_config = CollectionConfig {
            num_features,
            ..CollectionConfig::default_test_config()
        };
        let multi_spann_builder = MultiSpannBuilder::new(collection_config, base_dir.to_string()).unwrap();

        let doc_ids = [1u128, 2, 3, 4, 5];
        let point_ids: Vec<u32> = doc_ids.iter().map(|&doc_id| {
            multi_spann_builder.insert(user_id, doc_id, &generate_random_vector(num_features)).unwrap()
        }).collect();

        let multi_builder = MultiTermBuilder::new();
        point_ids.iter().for_each(|&pid| {
            multi_builder.add(user_id, pid, format!("field:term{}", pid)).unwrap();
        });
        multi_builder.build().unwrap();

        let multi_writer = MultiTermWriter::new(term_dir.clone());
        multi_writer.write(&multi_builder).unwrap();

        (Arc::new(MultiTermIndex::new(term_dir).unwrap()), user_id, format!("{}/terms/combined", base_dir))
    }

    #[tokio::test]
    async fn test_async_planner_basic() {
        let temp_dir = TempDir::new("async_planner_test").unwrap();
        let (multi_term_index, user_id, _) = create_test_indexes(&temp_dir).await;

        let ids_filter = IdsFilter { ids: vec![1, 3, 5] };
        let document_filter = DocumentFilter {
            filter: Some(proto::muopdb::document_filter::Filter::Ids(ids_filter)),
        };

        let planner = AsyncPlanner::new(user_id, document_filter, multi_term_index, None).await.unwrap();
        let mut iter = planner.plan().await.unwrap();

        assert_eq!(iter.next().await.unwrap(), Some(1));
        assert_eq!(iter.next().await.unwrap(), Some(3));
        assert_eq!(iter.next().await.unwrap(), Some(5));
        assert_eq!(iter.next().await.unwrap(), None);
    }
}
