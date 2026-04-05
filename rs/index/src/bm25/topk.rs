use std::cmp::Reverse;
use std::collections::BinaryHeap;

use ordered_float::OrderedFloat;

/// A generic Top-K selector using a min-heap for efficient K-selection
///
/// This selector maintains the top K highest-scoring items using a min-heap,
/// providing O(N log K) performance for selecting the top K items from N candidates.
///
/// The min-heap stores items with their scores using `Reverse` ordering,
/// so the smallest score is always at the top. This allows O(1) access to
/// the current K-th highest score.
///
/// # Type Parameters
/// * `T` - The item type being selected (must implement Clone + Ord for tie-breaking)
///
/// # Example
/// ```
/// use index::bm25::TopKSelector;
///
/// let mut selector = TopKSelector::new(3);
/// selector.offer("doc1", 10.0);
/// selector.offer("doc2", 50.0);
/// selector.offer("doc3", 30.0);
/// selector.offer("doc4", 40.0);
/// selector.offer("doc5", 20.0);
///
/// let top3 = selector.into_sorted_vec();
/// assert_eq!(top3, vec!["doc2", "doc4", "doc3"]);
/// ```
pub struct TopKSelector<T> {
    /// Maximum number of items to select
    k: usize,
    /// Min-heap storing (score, item) pairs
    /// Uses Reverse to make BinaryHeap a min-heap (smallest on top)
    /// Uses OrderedFloat for f32 to satisfy Ord requirement
    /// Uses Reverse<T> for the item to satisfy Ord when T: Ord
    heap: BinaryHeap<Reverse<(OrderedFloat<f32>, Reverse<T>)>>,
}

impl<T: Clone + Ord> TopKSelector<T> {
    /// Create a new TopKSelector with capacity K
    ///
    /// # Arguments
    /// * `k` - Maximum number of top items to select
    ///
    /// # Returns
    /// A new TopKSelector ready to accept items
    pub fn new(k: usize) -> Self {
        Self {
            k,
            heap: BinaryHeap::with_capacity(k),
        }
    }

    /// Offer an item with a score to the selector
    ///
    /// If the heap has fewer than K items, the item is added.
    /// If the heap is full (K items) and the score is higher than the
    /// current minimum, the minimum is evicted and the new item is added.
    ///
    /// This maintains exactly K highest-scoring items seen so far.
    ///
    /// # Arguments
    /// * `item` - The item to consider
    /// * `score` - The score associated with the item (higher = better)
    pub fn offer(&mut self, item: T, score: f32) {
        if self.k == 0 {
            return;
        }

        let ordered_score = OrderedFloat(score);

        if self.heap.len() < self.k {
            // Heap not full, just push
            self.heap.push(Reverse((ordered_score, Reverse(item))));
        } else if let Some(&Reverse((min_score, _))) = self.heap.peek() {
            // Heap is full, only add if score beats the minimum
            if ordered_score > min_score {
                self.heap.pop(); // Remove the minimum
                self.heap.push(Reverse((ordered_score, Reverse(item))));
            }
        }
    }

    /// Consume the selector and return the top K items sorted by score descending
    ///
    /// Returns the items in rank order (1st, 2nd, 3rd, ...) with highest scores first.
    ///
    /// # Returns
    /// A vector containing up to K items, sorted by score descending
    pub fn into_sorted_vec(self) -> Vec<T> {
        // BinaryHeap::into_sorted_vec() returns elements in ascending order according to Ord.
        // Since we use Reverse<(score, item)>, ascending order means from smallest Reverse
        // to largest Reverse. Since Reverse reverses, smallest Reverse = largest score.
        // So we already get descending order by score naturally!
        self.heap
            .into_sorted_vec()
            .into_iter()
            .map(|Reverse((_, Reverse(item)))| item)
            .collect()
    }

    /// Get the current minimum score in the heap (K-th highest score)
    ///
    /// Returns None if the heap has fewer than K items.
    /// This can be used for early termination optimization.
    ///
    /// # Returns
    /// * `Some(f32)` - The minimum score currently in the heap
    /// * `None` - If heap has fewer than K items
    pub fn current_min_score(&self) -> Option<f32> {
        self.heap.peek().map(|Reverse((score, _))| score.0)
    }

    /// Get the current number of items in the selector
    ///
    /// # Returns
    /// The number of items currently stored (may be less than K)
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// Check if the selector is empty
    ///
    /// # Returns
    /// True if no items have been added
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Check if the selector is full (has K items)
    ///
    /// # Returns
    /// True if the heap contains K items
    pub fn is_full(&self) -> bool {
        self.heap.len() == self.k
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topk_basic_selection() {
        let mut selector = TopKSelector::new(3);

        // Offer 5 items with scores: 10, 20, 30, 40, 50
        selector.offer(1u32, 10.0);
        selector.offer(2, 20.0);
        selector.offer(3, 30.0);
        selector.offer(4, 40.0);
        selector.offer(5, 50.0);

        let top3 = selector.into_sorted_vec();

        // Should return top 3 in descending order: 5, 4, 3 (highest scores first)
        assert_eq!(top3, vec![5, 4, 3]);
    }

    #[test]
    fn test_topk_exactly_k_items() {
        let mut selector = TopKSelector::new(5);

        // Offer exactly 5 items
        selector.offer(1, 10.0);
        selector.offer(2, 20.0);
        selector.offer(3, 30.0);
        selector.offer(4, 40.0);
        selector.offer(5, 50.0);

        let results = selector.into_sorted_vec();

        // All 5 should be returned in correct descending order
        assert_eq!(results, vec![5, 4, 3, 2, 1]);
    }

    #[test]
    fn test_topk_fewer_than_k() {
        let mut selector = TopKSelector::new(10);

        // Offer only 3 items when K=10
        selector.offer(1, 10.0);
        selector.offer(2, 20.0);
        selector.offer(3, 30.0);

        let results = selector.into_sorted_vec();

        // All 3 should be returned in descending order
        assert_eq!(results.len(), 3);
        assert_eq!(results, vec![3, 2, 1]);
    }

    #[test]
    fn test_topk_empty() {
        let selector: TopKSelector<u32> = TopKSelector::new(5);

        let results = selector.into_sorted_vec();

        // Should return empty vec
        assert!(results.is_empty());
    }

    #[test]
    fn test_topk_tie_breaking() {
        let mut selector = TopKSelector::new(3);

        // Items with same score
        selector.offer(1, 50.0);
        selector.offer(2, 50.0);
        selector.offer(3, 50.0);
        selector.offer(4, 40.0);
        selector.offer(5, 40.0);

        let results = selector.into_sorted_vec();

        // Should have 3 items (all with score 50.0)
        assert_eq!(results.len(), 3);
        // With Reverse wrapper, higher items should come first when scores are equal
        // because Reverse(3) < Reverse(2) < Reverse(1)
        // All should have score 50, order among ties: higher id first
        let result_set: std::collections::HashSet<u32> = results.iter().copied().collect();
        assert!(result_set.contains(&1));
        assert!(result_set.contains(&2));
        assert!(result_set.contains(&3));
        assert!(!result_set.contains(&4));
        assert!(!result_set.contains(&5));
    }

    #[test]
    fn test_current_min_score() {
        let mut selector = TopKSelector::new(3);

        // Initially empty
        assert_eq!(selector.current_min_score(), None);

        // Add one item
        selector.offer(1, 10.0);
        assert_eq!(selector.current_min_score(), Some(10.0));

        // Add more items
        selector.offer(2, 20.0);
        selector.offer(3, 30.0);
        // Now full, min should be 10.0
        assert_eq!(selector.current_min_score(), Some(10.0));

        // Add higher score - should evict 10.0
        selector.offer(4, 40.0);
        // Min should now be 20.0
        assert_eq!(selector.current_min_score(), Some(20.0));
    }

    #[test]
    fn test_offer_higher_than_min() {
        let mut selector = TopKSelector::new(3);

        // Fill to capacity with low scores
        selector.offer(1, 10.0);
        selector.offer(2, 20.0);
        selector.offer(3, 30.0);

        let min_before = selector.current_min_score();
        assert_eq!(min_before, Some(10.0));

        // Offer higher score, should replace lowest
        selector.offer(4, 40.0);

        // Now top 3 should be: 40, 30, 20 (not 10)
        let results = selector.into_sorted_vec();
        assert_eq!(results, vec![4, 3, 2]);
        assert!(!results.contains(&1)); // 10 was evicted
    }

    #[test]
    fn test_len_and_is_empty() {
        let mut selector = TopKSelector::new(5);

        assert!(selector.is_empty());
        assert_eq!(selector.len(), 0);

        selector.offer(1, 10.0);
        assert!(!selector.is_empty());
        assert_eq!(selector.len(), 1);

        selector.offer(2, 20.0);
        assert_eq!(selector.len(), 2);
    }

    #[test]
    fn test_is_full() {
        let mut selector = TopKSelector::new(3);

        assert!(!selector.is_full());

        selector.offer(1, 10.0);
        assert!(!selector.is_full());

        selector.offer(2, 20.0);
        selector.offer(3, 30.0);
        assert!(selector.is_full());

        // Adding more when full should not increase size
        selector.offer(4, 40.0);
        assert_eq!(selector.len(), 3);
    }

    #[test]
    fn test_k_zero() {
        let mut selector: TopKSelector<u32> = TopKSelector::new(0);

        // Should not panic, just ignore all offers
        selector.offer(1, 100.0);
        selector.offer(2, 200.0);

        let results = selector.into_sorted_vec();
        assert!(results.is_empty());
    }

    #[test]
    fn test_descending_order() {
        let mut selector = TopKSelector::new(5);

        // Offer in random order
        selector.offer(1, 25.0);
        selector.offer(2, 10.0);
        selector.offer(3, 50.0);
        selector.offer(4, 15.0);
        selector.offer(5, 35.0);

        let results = selector.into_sorted_vec();

        // Should be sorted descending: 50, 35, 25, 15, 10
        // Items: 3, 5, 1, 4, 2
        assert_eq!(results, vec![3, 5, 1, 4, 2]);
    }

    #[test]
    fn test_negative_scores() {
        let mut selector = TopKSelector::new(3);

        // Offer items with negative scores
        selector.offer(1, -10.0);
        selector.offer(2, -5.0);
        selector.offer(3, -20.0);
        selector.offer(4, 0.0); // Zero is higher than negative
        selector.offer(5, -15.0);

        let results = selector.into_sorted_vec();

        // Top 3 should be: 0.0, -5.0, -10.0 (items 4, 2, 1)
        assert_eq!(results, vec![4, 2, 1]);
    }

    #[test]
    fn test_zero_score() {
        let mut selector = TopKSelector::new(3);

        selector.offer(1, 0.0);
        selector.offer(2, 10.0);
        selector.offer(3, 5.0);
        selector.offer(4, 0.0);
        selector.offer(5, 20.0);

        let results = selector.into_sorted_vec();

        // Top 3: 20, 10, 5 (items 5, 2, 3)
        assert_eq!(results, vec![5, 2, 3]);
    }

    #[test]
    fn test_large_k() {
        let mut selector = TopKSelector::new(100);

        // Add 50 items
        for i in 0..50 {
            selector.offer(i, i as f32);
        }

        let results = selector.into_sorted_vec();

        // Should have all 50 items
        assert_eq!(results.len(), 50);

        // Should be in descending order
        assert_eq!(results[0], 49);
        assert_eq!(results[49], 0);
    }

    #[test]
    fn test_with_bm25_search_results() {
        use crate::bm25::query::Bm25SearchResult;

        let mut selector = TopKSelector::new(3);

        // Simulate BM25 scores
        selector.offer(Bm25SearchResult::new(1, 1.5), 1.5);
        selector.offer(Bm25SearchResult::new(2, 2.3), 2.3);
        selector.offer(Bm25SearchResult::new(3, 0.8), 0.8);
        selector.offer(Bm25SearchResult::new(4, 3.1), 3.1);
        selector.offer(Bm25SearchResult::new(5, 1.9), 1.9);

        let results = selector.into_sorted_vec();

        // Should be sorted by score descending
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].point_id, 4); // score 3.1
        assert_eq!(results[1].point_id, 2); // score 2.3
        assert_eq!(results[2].point_id, 5); // score 1.9
    }
}
