use rkyv::util::AlignedVec;
use utils::mem::{transmute_u8_to_slice, transmute_u8_to_val};

#[derive(Debug, Clone)]
pub struct WalEntry {
    pub buffer: AlignedVec<16>,
    pub seq_no: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WalOpType {
    Insert,
    Delete,
}

pub struct WalEntryDecoded<'a> {
    pub doc_ids: &'a [u128],
    pub user_ids: &'a [u128],
    pub data: &'a [f32],
    pub op_type: WalOpType,
}

impl WalEntry {
    pub fn decode(&self, num_features: usize) -> WalEntryDecoded {
        let length = self.buffer.len() - 1;
        let mut offset = 0;
        let num_docs = transmute_u8_to_val::<u64>(&self.buffer[offset..offset + 8]) as usize;
        offset += 8;
        let num_users = transmute_u8_to_val::<u64>(&self.buffer[offset..offset + 8]) as usize;
        offset += 8;
        let doc_ids = transmute_u8_to_slice::<u128>(&self.buffer[offset..offset + num_docs * 16]);
        offset += num_docs * 16;
        let user_ids = transmute_u8_to_slice::<u128>(&self.buffer[offset..offset + num_users * 16]);
        offset += num_users * 16;
        let data = transmute_u8_to_slice::<f32>(&self.buffer[offset..length]);

        assert_eq!(
            data.len(),
            num_features * num_docs,
            "num_vectors mismatch while decoding WalEntry data"
        );

        let op_type = if self.buffer[length] == 0 {
            WalOpType::Insert
        } else {
            WalOpType::Delete
        };
        WalEntryDecoded {
            doc_ids,
            user_ids,
            data,
            op_type,
        }
    }
}
