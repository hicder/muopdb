use anyhow::Result;
use atomic_refcell::AtomicRefCell;
use parking_lot::RwLock;

use crate::segment::mutable_segment::MutableSegment;

#[derive(Debug, Clone)]
pub struct DeletionOps {
    pub user_id: u128,
    pub doc_id: u128,
}

pub struct PendingMutableSegment {
    mutable_segment: AtomicRefCell<MutableSegment>,
    deletion_ops: RwLock<Vec<DeletionOps>>,
    last_sequence_number: u64,
}

impl PendingMutableSegment {
    pub fn new(mutable_segment: MutableSegment) -> Self {
        let last_sequence_number = mutable_segment.last_sequence_number();

        Self {
            mutable_segment: AtomicRefCell::new(mutable_segment),
            deletion_ops: RwLock::new(Vec::new()),
            last_sequence_number,
        }
    }

    // Make absolutely sure that this function is called only once.
    pub fn build(&self, base_directory: String, name: String) -> Result<()> {
        self.mutable_segment
            .borrow_mut()
            .build(base_directory, name)
    }

    pub fn invalidate(&self, user_id: u128, doc_id: u128) -> Result<bool> {
        self.deletion_ops
            .write()
            .push(DeletionOps { user_id, doc_id });
        Ok(true)
    }

    pub fn last_sequence_number(&self) -> u64 {
        self.last_sequence_number
    }

    pub fn deletion_ops(&self) -> Vec<DeletionOps> {
        self.deletion_ops.read().clone()
    }
}
