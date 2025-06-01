use utils::on_disk_ordered_map::map::OnDiskOrderedMap;
use utils::on_disk_ordered_map::encoder::VarintIntegerCodec;
use std::sync::Arc;

pub struct TermIndex {
    term_map: OnDiskOrderedMap<'static, VarintIntegerCodec>,
    _mmap: Arc<memmap2::Mmap>,
    _backing_file: std::fs::File,
}

impl TermIndex {
    pub fn new(path: String) -> anyhow::Result<Self> {
        let backing_file = std::fs::File::open(&path)?;
        let mmap = Arc::new(unsafe { memmap2::Mmap::map(&backing_file)? });
        
        // Clone the Arc to extend the lifetime
        let mmap_for_map = Arc::clone(&mmap);
        
        // SAFETY: We're extending the lifetime to 'static because we know
        // the Arc<Mmap> will live as long as this struct
        let mmap_ref: &'static memmap2::Mmap = unsafe {
            std::mem::transmute(mmap_for_map.as_ref())
        };
        
        let term_map = OnDiskOrderedMap::<VarintIntegerCodec>::new(
            path, 
            mmap_ref, 
            0, 
            mmap_ref.len()
        )?;
        
        Ok(TermIndex {
            term_map,
            _mmap: mmap,
            _backing_file: backing_file,
        })
    }
    
    pub fn term_map(&self) -> &OnDiskOrderedMap<'static, VarintIntegerCodec> {
        &self.term_map
    }
}
