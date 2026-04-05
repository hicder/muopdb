use std::collections::HashMap;

use anyhow::Result;
use dashmap::DashMap;
use ordered_float::OrderedFloat;

use crate::bm25::config::BM25Config;
use crate::bm25::query::Bm25Query;
use crate::bm25::searcher::Bm25Searcher;
use crate::bm25::topk::TopKSelector;
use crate::segment::BoxedImmutableSegment;
use quantization::quantization::Quantizer;

/// Per-segment search context for BM25 text search
///
/// Holds the initialized Bm25Searcher for a single segment along with
/// the segment metadata needed for cross-segment aggregation.
pub struct SegmentBm25Context {
    /// Identifier for the segment
    pub segment_name: String,
    /// Path to the bm25_stats directory for this segment
    pub stats_dir: String,
    /// Path to the terms directory for this segment
    pub terms_dir: String,
    /// Initialized BM25 searcher for this segment
    pub searcher: Bm25Searcher,
}

impl SegmentBm25Context {
    /// Create a new segment context from segment paths
    ///
    /// # Arguments
    /// * `segment_name` - Identifier for the segment
    /// * `stats_dir` - Path to the bm25_stats directory
    /// * `terms_dir` - Path to the terms directory
    /// * `config` - BM25 scoring configuration
    ///
    /// # Returns
    /// A Result containing the initialized SegmentBm25Context
    pub fn new(
        segment_name: String,
        stats_dir: String,
        terms_dir: String,
        config: BM25Config,
    ) -> Result<Self> {
        let searcher = Bm25Searcher::new(stats_dir.clone(), terms_dir.clone(), config)?;
        Ok(Self {
            segment_name,
            stats_dir,
            terms_dir,
            searcher,
        })
    }
}

/// Information about which segment contributed to a result
#[derive(Debug, Clone)]
pub struct SegmentHit {
    /// Name of the segment that contributed
    pub segment_name: String,
    /// The internal point ID in that segment
    pub point_id: u32,
    /// Score from that segment
    pub score: f32,
}

/// A single cross-segment search result
///
/// Contains the external document ID, aggregated BM25 score,
/// and information about which segments contributed (for debugging).
#[derive(Debug, Clone)]
pub struct CrossSegmentResult {
    /// External document ID (unique across all segments)
    pub doc_id: u128,
    /// Aggregated BM25 score (highest score if document appears in multiple segments)
    pub score: f32,
    /// Which segments contributed to this result (for debugging/audit)
    pub segment_hits: Vec<SegmentHit>,
}

/// Container for cross-segment search results
///
/// Contains the aggregated results from searching multiple segments,
/// sorted by BM25 score descending.
#[derive(Debug, Clone)]
pub struct CrossSegmentResults {
    /// Results sorted by score descending
    pub results: Vec<CrossSegmentResult>,
    /// Number of segments that were searched
    pub total_segments_searched: usize,
    /// Total number of documents considered before deduplication
    pub total_docs_considered: usize,
}

impl CrossSegmentResults {
    /// Get the number of results
    pub fn len(&self) -> usize {
        self.results.len()
    }

    /// Check if results are empty
    pub fn is_empty(&self) -> bool {
        self.results.is_empty()
    }

    /// Get the top result
    pub fn top_result(&self) -> Option<&CrossSegmentResult> {
        self.results.first()
    }
}

/// Cross-segment text searcher for BM25 queries
///
/// Manages BM25 search across multiple segments in a collection,
/// aggregating results and handling deduplication when the same
/// document appears in multiple segments.
pub struct CrossSegmentTextSearcher {
    /// Contexts for all segments with BM25 data
    segments: Vec<SegmentBm25Context>,
    /// Global top-K to return
    global_k: usize,
}

impl CrossSegmentTextSearcher {
    /// Create a new cross-segment searcher
    ///
    /// # Arguments
    /// * `segment_dirs` - Vector of (name, stats_dir, terms_dir) tuples for each segment
    /// * `config` - BM25 scoring configuration
    /// * `global_k` - Number of top results to return
    ///
    /// # Returns
    /// A configured CrossSegmentTextSearcher ready to search
    pub fn new(
        segment_dirs: Vec<(String, String, String)>,
        config: BM25Config,
        global_k: usize,
    ) -> Result<Self> {
        let mut segments = Vec::with_capacity(segment_dirs.len());

        for (name, stats_dir, terms_dir) in segment_dirs {
            match SegmentBm25Context::new(name.clone(), stats_dir, terms_dir, config.clone()) {
                Ok(context) => segments.push(context),
                Err(e) => {
                    // Log the error but continue with other segments
                    // A segment might not have BM25 data (e.g., created before BM25 was enabled)
                    tracing::debug!(
                        "Skipping segment {} for text search (no BM25 data): {}",
                        name,
                        e
                    );
                }
            }
        }

        Ok(Self {
            segments,
            global_k,
        })
    }

    /// Search across all segments and return aggregated results
    ///
    /// This method:
    /// 1. Searches each segment independently
    /// 2. Converts point_ids to external doc_ids
    /// 3. Deduplicates by doc_id (keeping highest score)
    /// 4. Selects global top-K using TopKSelector
    ///
    /// # Arguments
    /// * `query_text` - The search query text
    /// * `user_id` - Which user's documents to search
    /// * `all_segments` - Reference to the collection's segments (for doc_id lookup)
    ///
    /// # Returns
    /// CrossSegmentResults containing the aggregated top-K results
    pub async fn search<Q: Quantizer + Clone + Send + Sync>(
        &self,
        query_text: &str,
        user_id: u128,
        all_segments: &DashMap<String, BoxedImmutableSegment<Q>>,
    ) -> Result<CrossSegmentResults> {
        let mut total_docs_considered = 0usize;

        // HashMap to aggregate scores by doc_id
        // Key: doc_id, Value: (score, Vec of segment hits)
        let mut doc_aggregates: HashMap<u128, (f32, Vec<SegmentHit>)> = HashMap::new();

        // Search each segment
        for segment_context in &self.segments {
            let query = Bm25Query::with_max_results(
                query_text.to_string(),
                user_id,
                self.global_k * 2, // Request more than K to account for deduplication
            );

            let segment_results = match segment_context.searcher.search(&query) {
                Ok(results) => results,
                Err(e) => {
                    tracing::debug!(
                        "Segment {} search failed: {}",
                        segment_context.segment_name,
                        e
                    );
                    continue;
                }
            };

            total_docs_considered += segment_results.total_hits;

            // Get the segment from all_segments for doc_id lookup
            let segment = match all_segments.get(&segment_context.segment_name) {
                Some(seg) => seg.clone(),
                None => {
                    tracing::warn!(
                        "Segment {} not found in all_segments",
                        segment_context.segment_name
                    );
                    continue;
                }
            };

            // Convert point_ids to doc_ids and aggregate
            for result in &segment_results.results {
                let doc_id = match segment.get_doc_id(user_id, result.point_id).await {
                    Some(id) => id,
                    None => {
                        tracing::debug!(
                            "No doc_id found for user {} point {} in segment {}",
                            user_id,
                            result.point_id,
                            segment_context.segment_name
                        );
                        continue;
                    }
                };

                let segment_hit = SegmentHit {
                    segment_name: segment_context.segment_name.clone(),
                    point_id: result.point_id,
                    score: result.score,
                };

                // Aggregate by doc_id, keeping highest score
                doc_aggregates
                    .entry(doc_id)
                    .and_modify(|(score, hits)| {
                        if result.score > *score {
                            *score = result.score;
                        }
                        hits.push(segment_hit.clone());
                    })
                    .or_insert((result.score, vec![segment_hit]));
            }
        }

        // Select global top-K using TopKSelector
        // We use (doc_id, score) tuples with OrderedFloat for proper ordering
        let mut selector: TopKSelector<(u128, OrderedFloat<f32>)> =
            TopKSelector::new(self.global_k);

        for (doc_id, (score, _hits)) in &doc_aggregates {
            selector.offer((*doc_id, OrderedFloat(*score)), *score);
        }

        // Convert to final results
        let sorted_items = selector.into_sorted_vec();
        
        // Reconstruct the full results with segment hits
        let mut results: Vec<CrossSegmentResult> = Vec::with_capacity(sorted_items.len());
        for (doc_id, score) in sorted_items {
            // Recover the segment_hits from our aggregates
            if let Some((_, hits)) = doc_aggregates.get(&doc_id) {
                results.push(CrossSegmentResult {
                    doc_id,
                    score: score.0,
                    segment_hits: hits.clone(),
                });
            }
        }

        Ok(CrossSegmentResults {
            results,
            total_segments_searched: self.segments.len(),
            total_docs_considered,
        })
    }

    /// Get the number of segments being searched
    pub fn num_segments(&self) -> usize {
        self.segments.len()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use tempdir::TempDir;

    use super::*;
    use crate::bm25::stats_collector::MultiBm25StatsCollector;
    use crate::bm25::stats_writer::Bm25StatsWriter;
    use crate::multi_terms::builder::MultiTermBuilder;
    use crate::multi_terms::writer::MultiTermWriter;

    /// Helper to create a segment with BM25 data
    fn create_test_segment(
        base_dir: &str,
        segment_name: &str,
        user_id: u128,
        docs: Vec<(u32, Vec<String>)>,
    ) -> Result<(String, String, String)> {
        let segment_dir = format!("{}/{}", base_dir, segment_name);
        std::fs::create_dir_all(&segment_dir)?;

        let stats_dir = format!("{}/bm25_stats", segment_dir);
        let terms_dir = format!("{}/terms", segment_dir);
        std::fs::create_dir_all(&stats_dir)?;
        std::fs::create_dir_all(&terms_dir)?;

        // Build stats
        let collector = MultiBm25StatsCollector::new();
        let mut max_point_id: u32 = 0;

        for (point_id, terms) in &docs {
            collector.add_document(user_id, *point_id, terms.clone())?;
            if *point_id > max_point_id {
                max_point_id = *point_id;
            }
        }

        let mut max_point_ids: HashMap<u128, u32> = HashMap::new();
        max_point_ids.insert(user_id, max_point_id);

        let stats_writer = Bm25StatsWriter::new(segment_dir.clone());
        stats_writer.write(&collector, &max_point_ids)?;

        // Build term index
        let multi_builder = MultiTermBuilder::new();
        for (point_id, terms) in &docs {
            for term in terms {
                multi_builder.add(user_id, *point_id, term.clone())?;
            }
        }
        multi_builder.build()?;

        let writer = MultiTermWriter::new(terms_dir.clone());
        writer.write(&multi_builder)?;

        Ok((segment_name.to_string(), segment_dir, terms_dir))
    }

    #[test]
    fn test_cross_segment_search_basic() {
        let tmp_dir = TempDir::new("test_cross_segment").unwrap();
        let base_dir = tmp_dir.path().to_str().unwrap().to_string();
        let user_id = 123u128;

        // Create segment 1: Doc A (contains "apple"), Doc B (contains "banana")
        let docs1 = vec![
            (0, vec!["apple".to_string()]),
            (1, vec!["banana".to_string()]),
        ];
        let (name1, stats1, terms1) = create_test_segment(&base_dir, "segment1", user_id, docs1).unwrap();

        // Create segment 2: Doc C (contains "apple"), Doc D (contains "cherry")
        let docs2 = vec![
            (0, vec!["apple".to_string()]),
            (1, vec!["cherry".to_string()]),
        ];
        let (name2, stats2, terms2) = create_test_segment(&base_dir, "segment2", user_id, docs2).unwrap();

        // Build searcher
        let segment_dirs = vec![
            (name1, stats1, terms1),
            (name2, stats2, terms2),
        ];
        let searcher = CrossSegmentTextSearcher::new(
            segment_dirs,
            BM25Config::default(),
            10,
        ).unwrap();

        // Search for "apple"
        // Note: This is a simplified test - we can't easily provide all_segments DashMap
        // In a real scenario, this would use actual segments with doc_id mappings
        // For now, we just verify the searcher was created successfully
        assert_eq!(searcher.segments.len(), 2);
    }

    #[test]
    fn test_segment_context_creation() {
        let tmp_dir = TempDir::new("test_context").unwrap();
        let base_dir = tmp_dir.path().to_str().unwrap().to_string();
        let user_id = 123u128;

        // Create a segment with BM25 data
        let docs = vec![
            (0, vec!["apple".to_string(), "banana".to_string()]),
            (1, vec!["apple".to_string(), "cherry".to_string()]),
        ];
        let (name, stats_dir, terms_dir) = create_test_segment(&base_dir, "segment1", user_id, docs).unwrap();

        // Create context
        let context = SegmentBm25Context::new(name, stats_dir, terms_dir, BM25Config::default());
        assert!(context.is_ok());
    }

    #[test]
    fn test_cross_segment_results() {
        let results = CrossSegmentResults {
            results: vec![
                CrossSegmentResult {
                    doc_id: 1u128,
                    score: 10.0,
                    segment_hits: vec![
                        SegmentHit {
                            segment_name: "seg1".to_string(),
                            point_id: 0,
                            score: 10.0,
                        },
                    ],
                },
                CrossSegmentResult {
                    doc_id: 2u128,
                    score: 5.0,
                    segment_hits: vec![
                        SegmentHit {
                            segment_name: "seg2".to_string(),
                            point_id: 1,
                            score: 5.0,
                        },
                    ],
                },
            ],
            total_segments_searched: 2,
            total_docs_considered: 5,
        };

        assert_eq!(results.len(), 2);
        assert!(!results.is_empty());
        assert_eq!(results.top_result().unwrap().doc_id, 1);
        assert_eq!(results.total_segments_searched, 2);
        assert_eq!(results.total_docs_considered, 5);
    }

    #[test]
    fn test_cross_segment_empty_results() {
        let results = CrossSegmentResults {
            results: vec![],
            total_segments_searched: 2,
            total_docs_considered: 0,
        };

        assert!(results.is_empty());
        assert_eq!(results.len(), 0);
        assert!(results.top_result().is_none());
    }

    #[test]
    fn test_segment_hit() {
        let hit = SegmentHit {
            segment_name: "segment1".to_string(),
            point_id: 42,
            score: 3.14,
        };

        assert_eq!(hit.segment_name, "segment1");
        assert_eq!(hit.point_id, 42);
        assert_eq!(hit.score, 3.14);
    }

    #[test]
    fn test_cross_segment_result() {
        let result = CrossSegmentResult {
            doc_id: 123u128,
            score: 15.5,
            segment_hits: vec![
                SegmentHit {
                    segment_name: "seg1".to_string(),
                    point_id: 0,
                    score: 15.5,
                },
            ],
        };

        assert_eq!(result.doc_id, 123);
        assert_eq!(result.score, 15.5);
        assert_eq!(result.segment_hits.len(), 1);
    }

    #[test]
    fn test_searcher_creation_empty() {
        let searcher = CrossSegmentTextSearcher::new(vec![], BM25Config::default(), 10).unwrap();
        assert_eq!(searcher.segments.len(), 0);
    }

    #[test]
    fn test_searcher_skips_missing_segments() {
        let tmp_dir = TempDir::new("test_skip").unwrap();
        let base_dir = tmp_dir.path().to_str().unwrap().to_string();
        let user_id = 123u128;

        // Create one valid segment
        let docs = vec![(0, vec!["apple".to_string()])];
        let (name1, stats1, terms1) = create_test_segment(&base_dir, "valid", user_id, docs).unwrap();

        // Add invalid segment paths
        let invalid_stats = format!("{}/invalid/bm25_stats", base_dir);
        let invalid_terms = format!("{}/invalid/terms", base_dir);

        let segment_dirs = vec![
            (name1, stats1, terms1),
            ("invalid".to_string(), invalid_stats, invalid_terms),
        ];

        let searcher = CrossSegmentTextSearcher::new(segment_dirs, BM25Config::default(), 10).unwrap();
        // Should only have 1 valid segment
        assert_eq!(searcher.segments.len(), 1);
    }
}
