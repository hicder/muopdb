#[derive(Debug, Clone)]
pub struct SearchParams {
    pub top_k: usize,
    pub ef_construction: u32,
    pub record_pages: bool,
    pub num_explored_centroids: Option<usize>,
    pub centroid_distance_ratio: f32,
}

impl SearchParams {
    pub fn new(top_k: usize, ef_construction: u32, record_pages: bool) -> Self {
        Self {
            top_k,
            ef_construction,
            record_pages,
            num_explored_centroids: None,
            centroid_distance_ratio: 0.1,
        }
    }

    pub fn num_explored_centroids(&self) -> usize {
        self.num_explored_centroids.unwrap_or(self.top_k)
    }

    pub fn with_num_explored_centroids(mut self, num_explored_centroids: Option<usize>) -> Self {
        self.num_explored_centroids = num_explored_centroids;
        self
    }

    pub fn with_centroid_distance_ratio(mut self, ratio: f32) -> Self {
        self.centroid_distance_ratio = ratio;
        self
    }
}
