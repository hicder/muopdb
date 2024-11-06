use std::collections::HashSet;

use roaring::RoaringBitmap;

pub struct SearchContext {
    pub visited: RoaringBitmap,
    pub record_pages: bool,
    pub visited_pages: Option<HashSet<String>>,
}

impl SearchContext {
    pub fn new(record_pages: bool) -> Self {
        if !record_pages {
            Self {
                visited: RoaringBitmap::new(),
                record_pages: false,
                visited_pages: None,
            }
        } else {
            Self {
                visited: RoaringBitmap::new(),
                record_pages: true,
                visited_pages: Some(HashSet::new()),
            }
        }
    }

    pub fn num_pages_accessed(&self) -> usize {
        if !self.record_pages {
            return 0;
        }

        self.visited_pages.as_ref().unwrap().len()
    }
}
