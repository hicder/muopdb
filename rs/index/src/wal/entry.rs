use utils::mem::transmute_u8_to_slice;

#[derive(Debug, Clone)]
pub struct WalEntry {
    pub buffer: Vec<u8>,
    pub seq_no: u64,
}

pub struct WalEntryDecoded<'a> {
    pub doc_ids: &'a [u128],
    pub user_ids: &'a [u128],
    pub data: &'a [f32],
}

impl WalEntry {
    pub fn decode(&self, num_features: usize) -> WalEntryDecoded {
        let length = self.buffer.len();
        let num_vectors = length / (16 + 16 + 4 * num_features);
        let doc_ids = transmute_u8_to_slice::<u128>(&self.buffer[0..num_vectors * 16]);
        let user_ids = transmute_u8_to_slice::<u128>(
            &self.buffer[num_vectors * 16..num_vectors * 16 + num_vectors * 16],
        );
        let data =
            transmute_u8_to_slice::<f32>(&self.buffer[num_vectors * 16 + num_vectors * 16..]);
        WalEntryDecoded {
            doc_ids,
            user_ids,
            data,
        }
    }
}
