use utils::mem::{transmute_u8_to_slice, transmute_u8_to_val};

#[derive(Debug, Clone)]
pub struct WalEntry {
    pub buffer: Vec<u8>,
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
        let num_docs = transmute_u8_to_val::<u32>(&self.buffer[0..4]) as usize;
        let num_users = transmute_u8_to_val::<u32>(&self.buffer[4..8]) as usize;
        let doc_ids = transmute_u8_to_slice::<u128>(&self.buffer[8..8 + num_docs * 16]);
        let user_ids = transmute_u8_to_slice::<u128>(
            &self.buffer[8 + num_docs * 16..8 + num_docs * 16 + num_users * 16],
        );
        let data =
            transmute_u8_to_slice::<f32>(&self.buffer[8 + num_docs * 16 + num_users * 16..length]);

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
