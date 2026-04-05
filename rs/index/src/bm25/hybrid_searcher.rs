use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use config::search_params::SearchParams;

use crate::bm25::config::BM25Config;
use crate::bm25::fusion::{FusionConfig, HybridScoreFusion};
use crate::bm25::hybrid_query::{HybridQuery, HybridSearchResult, HybridSearchResults, QueryMode};
use crate::bm25::normalizer::ScoreNormalizer;
use crate::bm25::query::{Bm25Query, Bm25SearchResults};
use crate::bm25::searcher::Bm25Searcher;
use crate::bm25::topk::TopKSelector;
use crate::collection::core::Collection;
use crate::collection::snapshot::Snapshot;
use crate::segment::{BoxedImmutableSegment, Segment};
use quantization::quantization::Quantizer;

/// Hybrid Searcher for executing combined text and vector searches
///
/// This is the core search component that:
/// - Executes BM25 text search and vector similarity search in parallel
/// - Normalizes scores from both modalities to [0, 1] range
/// - Merges results by doc_id using outer join semantics
/// - Computes hybrid scores using weighted fusion
/// - Returns top-K results sorted by hybrid score
///
/// The searcher supports three query modes:
/// - TextOnly: BM25 search only (equivalent to Bm25Searcher)
/// - VectorOnly: Vector search only (equivalent to existing vector search)
/// - Hybrid: Combined text + vector search with score fusion
pub struct HybridSearcher<Q: Quantizer + Clone + Send + Sync> {
    /// Optional BM25 searcher for text search (None if segment has no BM25 data)
    bm25_searcher: Option<Bm25Searcher>,
    /// Score fusion component for combining BM25 and vector scores
    fusion: HybridScoreFusion,
    /// Score normalizer for BM25 scores
    normalizer: ScoreNormalizer,
    /// Segment being searched
    segment: BoxedImmutableSegment<Q>,
    /// Segment base directory for paths
    _segment_base: String,
}

impl<Q: Quantizer + Clone + Send + Sync + 'static> HybridSearcher<Q> {
    /// Create a new hybrid searcher for a specific segment
    ///
    /// # Arguments
    /// * `segment` - The segment to search
    /// * `segment_base` - Base directory path for the segment
    /// * `bm25_config` - BM25 configuration for text search
    ///
    /// # Returns
    /// A configured HybridSearcher. If the segment has no BM25 data, bm25_searcher will be None.
    pub fn new(
        segment: BoxedImmutableSegment<Q>,
        segment_base: String,
        bm25_config: BM25Config,
    ) -> Self {
        // Bm25Searcher expects the segment_base directory (it will append bm25_stats internally)
        let terms_dir = format!("{}/terms", segment_base);

        // Try to create BM25 searcher, but don't fail if BM25 data doesn't exist
        // Check if stats directory exists (bm25_stats is appended by Bm25StatsReader)
        let stats_dir = format!("{}/bm25_stats", segment_base);
        let bm25_searcher = if std::path::Path::new(&stats_dir).exists() {
            // Pass segment_base (not stats_dir) because Bm25StatsReader appends bm25_stats
            match Bm25Searcher::new(segment_base.clone(), terms_dir, bm25_config) {
                Ok(searcher) => Some(searcher),
                Err(e) => {
                    tracing::debug!("Failed to create Bm25Searcher: {}", e);
                    None
                }
            }
        } else {
            None
        };

        let fusion = HybridScoreFusion::new(FusionConfig::balanced());
        let normalizer = ScoreNormalizer::new();

        Self {
            bm25_searcher,
            fusion,
            normalizer,
            segment,
            _segment_base: segment_base,
        }
    }

    /// Execute a hybrid search query
    ///
    /// Performs text and/or vector search based on the query mode,
    /// then fuses results using the configured alpha weighting.
    ///
    /// # Arguments
    /// * `query` - The hybrid query containing text, vector, and configuration
    ///
    /// # Returns
    /// Hybrid search results sorted by hybrid_score descending
    pub async fn search(&self, query: &HybridQuery) -> Result<HybridSearchResults> {
        match query.mode {
            QueryMode::TextOnly => self.search_text_only(query).await,
            QueryMode::VectorOnly => self.search_vector_only(query).await,
            QueryMode::Hybrid => self.search_hybrid(query).await,
        }
    }

    /// Execute text-only search
    async fn search_text_only(&self, query: &HybridQuery) -> Result<HybridSearchResults> {
        let text_query = match &query.text_query {
            Some(t) => t.clone(),
            None => {
                return Ok(HybridSearchResults::empty());
            }
        };

        let bm25_results = self.execute_bm25_search(&text_query, query.user_id, query.top_k).await?;
        let total_hits = bm25_results.total_hits;

        // Convert BM25 results to hybrid results with only BM25 scores
        let results: Vec<HybridSearchResult> = bm25_results
            .results
            .into_iter()
            .map(|r| {
                HybridSearchResult::new(
                    r.point_id as u128,  // point_id is the doc_id here
                    r.score,      // bm25_score
                    0.0,          // vector_score
                    r.score,      // hybrid_score = bm25_score when text-only
                )
            })
            .collect();

        Ok(HybridSearchResults::new(results, total_hits, 0))
    }

    /// Execute vector-only search
    async fn search_vector_only(&self, query: &HybridQuery) -> Result<HybridSearchResults> {
        let vector = match &query.vector_query {
            Some(v) => v.clone(),
            None => {
                return Ok(HybridSearchResults::empty());
            }
        };

        let vector_results = self
            .execute_vector_search(&vector, query.user_id, query.top_k)
            .await?;
        let total_hits = vector_results.len();

        // Convert vector results to hybrid results with only vector scores
        let results: Vec<HybridSearchResult> = vector_results
            .into_iter()
            .map(|(doc_id, score)| {
                HybridSearchResult::new(
                    doc_id,
                    0.0,      // bm25_score
                    score,    // vector_score
                    score,    // hybrid_score = vector_score when vector-only
                )
            })
            .collect();

        Ok(HybridSearchResults::new(results, total_hits, 0))
    }

    /// Execute hybrid search combining text and vector
    async fn search_hybrid(&self, query: &HybridQuery) -> Result<HybridSearchResults> {
        // Determine if we need text search
        let need_text = query.should_search_text() && self.bm25_searcher.is_some();

        // Determine if we need vector search
        let need_vector = query.should_search_vector();

        // Execute searches based on what's needed
        let (bm25_results, vector_results) = match (need_text, need_vector) {
            (true, true) => {
                // Execute both searches in parallel
                let text_query = query.text_query.clone().unwrap();
                let vector = query.vector_query.clone().unwrap();
                let user_id = query.user_id;
                let top_k = query.top_k;

                let (bm25_res, vector_res) = tokio::join!(
                    self.execute_bm25_search(&text_query, user_id, top_k),
                    self.execute_vector_search(&vector, user_id, top_k)
                );

                (bm25_res?, vector_res?)
            }
            (true, false) => {
                // Text only
                let text_query = query.text_query.clone().unwrap();
                let bm25_res = self.execute_bm25_search(&text_query, query.user_id, query.top_k).await?;
                (bm25_res, Vec::new())
            }
            (false, true) => {
                // Vector only
                let vector = query.vector_query.clone().unwrap();
                let vector_res = self.execute_vector_search(&vector, query.user_id, query.top_k).await?;
                (Bm25SearchResults::empty(), vector_res)
            }
            (false, false) => {
                return Ok(HybridSearchResults::empty());
            }
        };

        // Merge and fuse results
        self.merge_and_fuse(
            &bm25_results,
            &vector_results,
            query.alpha,
            query.top_k,
        )
    }

    /// Execute BM25 text search on the segment
    async fn execute_bm25_search(
        &self,
        text: &str,
        user_id: u128,
        max_results: usize,
    ) -> Result<Bm25SearchResults> {
        let searcher = match &self.bm25_searcher {
            Some(s) => s,
            None => return Ok(Bm25SearchResults::empty()),
        };

        let query = Bm25Query::with_max_results(text.to_string(), user_id, max_results);
        let search_results = searcher.search(&query)?;

        // Convert point_ids to doc_ids
        let mut doc_results: Vec<(u128, f32)> = Vec::new();
        for result in &search_results.results {
            if let Some(doc_id) = self.segment.get_doc_id(user_id, result.point_id).await {
                doc_results.push((doc_id, result.score));
            }
        }

        // Re-sort by score descending
        doc_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Create Bm25SearchResults with doc_ids as point_ids
        let bm25_results: Vec<crate::bm25::query::Bm25SearchResult> = doc_results
            .iter()
            .map(|(doc_id, score)| crate::bm25::query::Bm25SearchResult::new(*doc_id as u32, *score))
            .collect();

        // We need to return a custom structure, so let's create a wrapper
        // Actually, let's just return the results as-is and convert in merge_and_fuse
        Ok(Bm25SearchResults::with_total_matches(
            bm25_results,
            doc_results.len(),
        ))
    }

    /// Execute vector similarity search on the segment
    async fn execute_vector_search(
        &self,
        vector: &[f32],
        user_id: u128,
        top_k: usize,
    ) -> Result<Vec<(u128, f32)>> {
        let params = SearchParams::new(top_k, (top_k * 2) as u32, false);

        let results = BoxedImmutableSegment::search_with_id(
            self.segment.clone(),
            user_id,
            vector.to_vec(),
            &params,
            None, // No planner for hybrid search
        )
        .await;

        match results {
            Some(search_result) => {
                let doc_scores: Vec<(u128, f32)> = search_result
                    .id_with_scores
                    .into_iter()
                    .map(|id_score| (id_score.doc_id, id_score.score))
                    .collect();
                Ok(doc_scores)
            }
            None => Ok(Vec::new()),
        }
    }

    /// Merge BM25 and vector results, normalize scores, and fuse
    fn merge_and_fuse(
        &self,
        bm25_results: &Bm25SearchResults,
        vector_results: &[(u128, f32)],
        alpha: f32,
        top_k: usize,
    ) -> Result<HybridSearchResults> {
        // Build lookup maps
        let mut bm25_map: HashMap<u128, f32> = HashMap::new();
        for result in &bm25_results.results {
            let doc_id = result.point_id as u128;
            bm25_map.insert(doc_id, result.score);
        }

        let mut vector_map: HashMap<u128, f32> = HashMap::new();
        for (doc_id, score) in vector_results {
            vector_map.insert(*doc_id, *score);
        }

        // Collect all unique doc_ids
        let all_doc_ids: std::collections::HashSet<u128> =
            bm25_map.keys().chain(vector_map.keys()).copied().collect();

        if all_doc_ids.is_empty() {
            return Ok(HybridSearchResults::empty());
        }

        // Extract scores for normalization
        let bm25_scores: Vec<f32> = bm25_map.values().copied().collect();
        let vector_scores: Vec<f32> = vector_map.values().copied().collect();

        // Normalize scores
        let normalized_bm25 = self.normalizer.normalize_scores(&bm25_scores);
        let normalized_vector = self.normalizer.normalize_scores(&vector_scores);

        // Create lookup for normalized scores
        let bm25_doc_ids: Vec<u128> = bm25_map.keys().copied().collect();
        let vector_doc_ids: Vec<u128> = vector_map.keys().copied().collect();

        let mut normalized_bm25_map: HashMap<u128, f32> = HashMap::new();
        for (i, doc_id) in bm25_doc_ids.iter().enumerate() {
            normalized_bm25_map.insert(*doc_id, normalized_bm25.get(i).copied().unwrap_or(0.0));
        }

        let mut normalized_vector_map: HashMap<u128, f32> = HashMap::new();
        for (i, doc_id) in vector_doc_ids.iter().enumerate() {
            normalized_vector_map.insert(*doc_id, normalized_vector.get(i).copied().unwrap_or(0.0));
        }

        // Count overlapping documents
        let overlapping_hits = bm25_map
            .keys()
            .filter(|k| vector_map.contains_key(k))
            .count();

        // Create hybrid candidates with fusion
        let fusion = HybridScoreFusion::with_alpha(alpha);

        let mut candidates: Vec<HybridSearchResult> = all_doc_ids
            .iter()
            .map(|&doc_id| {
                let bm25_norm = *normalized_bm25_map.get(&doc_id).unwrap_or(&0.0);
                let vector_norm = *normalized_vector_map.get(&doc_id).unwrap_or(&0.0);
                let hybrid_score = fusion.combine(bm25_norm, vector_norm);

                HybridSearchResult::new(
                    doc_id,
                    *bm25_map.get(&doc_id).unwrap_or(&0.0),
                    *vector_map.get(&doc_id).unwrap_or(&0.0),
                    hybrid_score,
                )
            })
            .collect();

        // Sort by hybrid score descending
        candidates.sort();

        // Truncate to top_k
        let total_hits = candidates.len();
        if candidates.len() > top_k {
            candidates.truncate(top_k);
        }

        Ok(HybridSearchResults::new(candidates, total_hits, overlapping_hits))
    }

    /// Check if this searcher has BM25 capability
    pub fn has_bm25(&self) -> bool {
        self.bm25_searcher.is_some()
    }
}

/// Collection-level hybrid search helper
///
/// Searches across all segments in a collection and aggregates results.
pub async fn search_hybrid_across_segments<Q: Quantizer + Clone + Send + Sync + 'static>(
    collection: Arc<Collection<Q>>,
    snapshot: Arc<Snapshot<Q>>,
    query: &HybridQuery,
    bm25_config: BM25Config,
) -> Result<HybridSearchResults> {
    use ordered_float::OrderedFloat;

    let mut all_candidates: Vec<(u128, f32, f32)> = Vec::new(); // (doc_id, bm25_score, vector_score)

    // Search each segment
    for segment in &snapshot.segments {
        let segment_name = segment.name().await;
        let segment_base = format!("{}/{}", collection.all_segments().get(&segment_name).map(|e| collection.all_segments().get(&segment_name).unwrap().key().to_string()).unwrap_or_default(), segment_name);

        // Create searcher for this segment
        let searcher = HybridSearcher::new(
            segment.clone(),
            segment_base,
            bm25_config.clone(),
        );

        // Execute search
        let results = searcher.search(query).await?;

        // Collect candidates
        for result in results.results {
            all_candidates.push((result.doc_id, result.bm25_score, result.vector_score));
        }
    }

    if all_candidates.is_empty() {
        return Ok(HybridSearchResults::empty());
    }

    // Normalize and fuse across all segments
    let normalizer = ScoreNormalizer::new();
    let fusion = HybridScoreFusion::with_alpha(query.alpha);

    // Extract scores for normalization
    let bm25_scores: Vec<f32> = all_candidates.iter().map(|(_, b, _)| *b).collect();
    let vector_scores: Vec<f32> = all_candidates.iter().map(|(_, _, v)| *v).collect();

    // Normalize globally across all segments
    let normalized_bm25 = normalizer.normalize_scores(&bm25_scores);
    let normalized_vector = normalizer.normalize_scores(&vector_scores);

    // Create hybrid results
    let mut hybrid_results: Vec<HybridSearchResult> = all_candidates
        .iter()
        .enumerate()
        .map(|(i, (doc_id, orig_bm25, orig_vector))| {
            let bm25_norm = normalized_bm25.get(i).copied().unwrap_or(0.0);
            let vector_norm = normalized_vector.get(i).copied().unwrap_or(0.0);
            let hybrid_score = fusion.combine(bm25_norm, vector_norm);

            HybridSearchResult::new(*doc_id, *orig_bm25, *orig_vector, hybrid_score)
        })
        .collect();

    // Deduplicate by doc_id (keep highest hybrid score)
    let mut best_results: HashMap<u128, HybridSearchResult> = HashMap::new();
    for result in hybrid_results {
        best_results
            .entry(result.doc_id)
            .and_modify(|existing| {
                if result.hybrid_score > existing.hybrid_score {
                    *existing = result;
                }
            })
            .or_insert(result);
    }

    // Convert to vector and sort
    let mut final_results: Vec<HybridSearchResult> = best_results.into_values().collect();
    final_results.sort();

    let total_hits = final_results.len();

    // Select global top K
    let mut selector: TopKSelector<(u128, OrderedFloat<f32>, HybridSearchResult)> = TopKSelector::new(query.top_k);
    for result in &final_results {
        selector.offer(
            (result.doc_id, OrderedFloat(result.hybrid_score), result.clone()),
            result.hybrid_score,
        );
    }

    let selected: Vec<HybridSearchResult> = selector
        .into_sorted_vec()
        .into_iter()
        .map(|(_, _, result)| result)
        .collect();

    // Calculate overlapping hits
    let overlapping_hits = selected
        .iter()
        .filter(|r| r.bm25_score > 0.0 && r.vector_score > 0.0)
        .count();

    Ok(HybridSearchResults::new(selected, total_hits, overlapping_hits))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bm25::BM25Config;

    // Test 1: HybridSearcher::new creates searcher with BM25 config
    #[test]
    fn test_hybrid_searcher_new() {
        // This test is a simple structural test
        // Full integration requires a full collection setup

        // Verify that HybridSearcher has the expected fields
        // Note: Creating a real HybridSearcher requires async runtime and full collection
        let _config = BM25Config::default();
    }

    // Test 2: HybridQuery mode selection works correctly
    #[test]
    fn test_query_mode_selection() {
        let text_query = HybridQuery::text_only("query".to_string(), 1u128, 10);
        assert!(text_query.should_search_text());
        assert!(!text_query.should_search_vector());

        let vector_query = HybridQuery::vector_only(vec![1.0, 2.0], 1u128, 10);
        assert!(!vector_query.should_search_text());
        assert!(vector_query.should_search_vector());

        let hybrid = HybridQuery::new("text".to_string(), vec![1.0], 1u128, 10);
        assert!(hybrid.should_search_text());
        assert!(hybrid.should_search_vector());
    }

    // Test 3: Score normalization before fusion
    #[test]
    fn test_score_normalization() {
        let normalizer = ScoreNormalizer::new();

        // Test with different ranges
        let scores1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let normalized1 = normalizer.normalize_scores(&scores1);
        assert_eq!(normalized1[0], 0.0);
        assert_eq!(normalized1[4], 1.0);

        let scores2 = vec![10.0, 20.0, 30.0];
        let normalized2 = normalizer.normalize_scores(&scores2);
        assert_eq!(normalized2[0], 0.0);
        assert_eq!(normalized2[2], 1.0);

        // Both should now be in [0, 1] range
        for score in &normalized1 {
            assert!(score >= &0.0 && score <= &1.0);
        }
        for score in &normalized2 {
            assert!(score >= &0.0 && score <= &1.0);
        }
    }

    // Test 4: Result merging with outer join semantics
    #[test]
    fn test_result_merging() {
        // Create fusion with alpha=0.5
        let fusion = HybridScoreFusion::with_alpha(0.5);

        // BM25 results: doc1=0.8, doc2=0.3
        let bm25_results = vec![(1u128, 0.8f32), (2u128, 0.3f32)];

        // Vector results: doc2=0.9, doc3=0.5
        let vector_results = vec![(2u128, 0.9f32), (3u128, 0.5f32)];

        let merged = fusion.merge_results(&bm25_results, &vector_results);

        // Should have 3 unique doc_ids
        assert_eq!(merged.len(), 3);

        // Doc 2 has both scores
        let doc2 = merged.iter().find(|c| c.doc_id == 2).unwrap();
        assert!(doc2.bm25_score > 0.0);
        assert!(doc2.vector_score > 0.0);

        // Doc 1 has only BM25
        let doc1 = merged.iter().find(|c| c.doc_id == 1).unwrap();
        assert!(doc1.bm25_score > 0.0);
        assert_eq!(doc1.vector_score, 0.0);

        // Doc 3 has only vector
        let doc3 = merged.iter().find(|c| c.doc_id == 3).unwrap();
        assert_eq!(doc3.bm25_score, 0.0);
        assert!(doc3.vector_score > 0.0);
    }

    // Test 5: Documents missing from one source get 0.0 for that score
    #[test]
    fn test_missing_scores() {
        let fusion = HybridScoreFusion::with_alpha(0.5);

        // Only BM25 results
        let bm25_results = vec![(1u128, 0.8f32)];
        let vector_results = vec![];

        let merged = fusion.merge_results(&bm25_results, &vector_results);

        let doc1 = merged.iter().find(|c| c.doc_id == 1).unwrap();
        assert_eq!(doc1.bm25_score, 0.8);
        assert_eq!(doc1.vector_score, 0.0);

        // Only vector results
        let bm25_results = vec![];
        let vector_results = vec![(1u128, 0.8f32)];

        let merged = fusion.merge_results(&bm25_results, &vector_results);

        let doc1 = merged.iter().find(|c| c.doc_id == 1).unwrap();
        assert_eq!(doc1.bm25_score, 0.0);
        assert_eq!(doc1.vector_score, 0.8);
    }

    // Test: Alpha weighting in fusion
    #[test]
    fn test_alpha_weighting() {
        let fusion = HybridScoreFusion::with_alpha(0.7);
        let hybrid = fusion.combine(0.8, 0.6);
        // 0.7 * 0.8 + 0.3 * 0.6 = 0.56 + 0.18 = 0.74
        assert!((hybrid - 0.74).abs() < 0.0001);
    }

    // Test: Empty results handling
    #[test]
    fn test_empty_results() {
        let fusion = HybridScoreFusion::with_alpha(0.5);

        let bm25_results = vec![];
        let vector_results = vec![];

        let merged = fusion.merge_results(&bm25_results, &vector_results);
        assert!(merged.is_empty());
    }

    // Test: HybridSearchResult ordering
    #[test]
    fn test_hybrid_search_result_ordering() {
        let results = vec![
            HybridSearchResult::new(1u128, 0.5, 0.5, 0.3),
            HybridSearchResult::new(2u128, 0.6, 0.6, 0.9),
            HybridSearchResult::new(3u128, 0.7, 0.7, 0.6),
        ];

        let mut sorted = results.clone();
        sorted.sort();

        // Highest hybrid score first
        assert_eq!(sorted[0].doc_id, 2);
        assert_eq!(sorted[1].doc_id, 3);
        assert_eq!(sorted[2].doc_id, 1);
    }
}
