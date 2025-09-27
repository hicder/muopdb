use odht::{Config, FxHashFn};

#[derive(Clone)]
pub struct UserIndexInfo {
    pub user_id: u128,
    pub centroid_vector_offset: u64,
    pub centroid_vector_len: u64,
    pub centroid_index_offset: u64,
    pub centroid_index_len: u64,
    pub ivf_vectors_offset: u64,
    pub ivf_vectors_len: u64,
    pub ivf_raw_vectors_offset: u64,
    pub ivf_raw_vectors_len: u64,
    pub ivf_index_offset: u64,
    pub ivf_index_len: u64,
    pub ivf_pq_codebook_offset: u64,
    pub ivf_pq_codebook_len: u64,
}

impl UserIndexInfo {
    #[inline]
    pub fn to_le_bytes(&self) -> [u8; 112] {
        let mut bytes = [0u8; 112];
        bytes[0..16].copy_from_slice(&self.user_id.to_le_bytes());
        bytes[16..24].copy_from_slice(&self.centroid_vector_offset.to_le_bytes());
        bytes[24..32].copy_from_slice(&self.centroid_vector_len.to_le_bytes());
        bytes[32..40].copy_from_slice(&self.centroid_index_offset.to_le_bytes());
        bytes[40..48].copy_from_slice(&self.centroid_index_len.to_le_bytes());
        bytes[48..56].copy_from_slice(&self.ivf_vectors_offset.to_le_bytes());
        bytes[56..64].copy_from_slice(&self.ivf_vectors_len.to_le_bytes());
        bytes[64..72].copy_from_slice(&self.ivf_raw_vectors_offset.to_le_bytes());
        bytes[72..80].copy_from_slice(&self.ivf_raw_vectors_len.to_le_bytes());
        bytes[80..88].copy_from_slice(&self.ivf_index_offset.to_le_bytes());
        bytes[88..96].copy_from_slice(&self.ivf_index_len.to_le_bytes());
        bytes[96..104].copy_from_slice(&self.ivf_pq_codebook_offset.to_le_bytes());
        bytes[104..112].copy_from_slice(&self.ivf_pq_codebook_len.to_le_bytes());
        bytes
    }

    #[inline]
    pub fn from_le_bytes(bytes: &[u8]) -> Self {
        let user_id = u128::from_le_bytes(bytes[0..16].try_into().unwrap());
        let centroid_vector_offset = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
        let centroid_vector_len = u64::from_le_bytes(bytes[24..32].try_into().unwrap());
        let centroid_index_offset = u64::from_le_bytes(bytes[32..40].try_into().unwrap());
        let centroid_index_len = u64::from_le_bytes(bytes[40..48].try_into().unwrap());
        let ivf_vectors_offset = u64::from_le_bytes(bytes[48..56].try_into().unwrap());
        let ivf_vectors_len = u64::from_le_bytes(bytes[56..64].try_into().unwrap());
        let ivf_raw_vectors_offset = u64::from_le_bytes(bytes[64..72].try_into().unwrap());
        let ivf_raw_vectors_len = u64::from_le_bytes(bytes[72..80].try_into().unwrap());
        let ivf_index_offset = u64::from_le_bytes(bytes[80..88].try_into().unwrap());
        let ivf_index_len = u64::from_le_bytes(bytes[88..96].try_into().unwrap());
        let ivf_pq_codebook_offset = u64::from_le_bytes(bytes[96..104].try_into().unwrap());
        let ivf_pq_codebook_len = u64::from_le_bytes(bytes[104..112].try_into().unwrap());
        Self {
            user_id,
            centroid_vector_offset,
            centroid_vector_len,
            centroid_index_offset,
            centroid_index_len,
            ivf_vectors_offset,
            ivf_vectors_len,
            ivf_raw_vectors_offset,
            ivf_raw_vectors_len,
            ivf_index_offset,
            ivf_index_len,
            ivf_pq_codebook_offset,
            ivf_pq_codebook_len,
        }
    }
}

pub struct HashConfig {}
impl Config for HashConfig {
    type Key = u128;
    type Value = UserIndexInfo;
    type EncodedKey = [u8; 16];
    type EncodedValue = [u8; 112];
    type H = FxHashFn;

    #[inline]
    fn encode_key(k: &Self::Key) -> Self::EncodedKey {
        k.to_le_bytes()
    }
    #[inline]
    fn encode_value(v: &Self::Value) -> Self::EncodedValue {
        v.to_le_bytes()
    }
    #[inline]
    fn decode_key(k: &Self::EncodedKey) -> Self::Key {
        u128::from_le_bytes(*k)
    }
    #[inline]
    fn decode_value(v: &Self::EncodedValue) -> Self::Value {
        UserIndexInfo::from_le_bytes(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_user_index_info_serialization() {
        let original_info = UserIndexInfo {
            user_id: 1234567890,
            centroid_vector_offset: 100,
            centroid_vector_len: 200,
            centroid_index_offset: 300,
            centroid_index_len: 400,
            ivf_vectors_offset: 500,
            ivf_vectors_len: 600,
            ivf_raw_vectors_offset: 700,
            ivf_raw_vectors_len: 800,
            ivf_index_offset: 900,
            ivf_index_len: 1000,
            ivf_pq_codebook_offset: 1100,
            ivf_pq_codebook_len: 1200,
        };

        let bytes = original_info.to_le_bytes();
        let deserialized_info = UserIndexInfo::from_le_bytes(&bytes);

        assert_eq!(original_info.user_id, deserialized_info.user_id);
        assert_eq!(
            original_info.centroid_vector_offset,
            deserialized_info.centroid_vector_offset
        );
        assert_eq!(
            original_info.centroid_vector_len,
            deserialized_info.centroid_vector_len
        );
        assert_eq!(
            original_info.centroid_index_offset,
            deserialized_info.centroid_index_offset
        );
        assert_eq!(
            original_info.centroid_index_len,
            deserialized_info.centroid_index_len
        );
        assert_eq!(
            original_info.ivf_vectors_offset,
            deserialized_info.ivf_vectors_offset
        );
        assert_eq!(
            original_info.ivf_vectors_len,
            deserialized_info.ivf_vectors_len
        );
        assert_eq!(
            original_info.ivf_raw_vectors_offset,
            deserialized_info.ivf_raw_vectors_offset
        );
        assert_eq!(
            original_info.ivf_raw_vectors_len,
            deserialized_info.ivf_raw_vectors_len
        );
        assert_eq!(
            original_info.ivf_index_offset,
            deserialized_info.ivf_index_offset
        );
        assert_eq!(original_info.ivf_index_len, deserialized_info.ivf_index_len);
        assert_eq!(
            original_info.ivf_pq_codebook_offset,
            deserialized_info.ivf_pq_codebook_offset
        );
        assert_eq!(
            original_info.ivf_pq_codebook_len,
            deserialized_info.ivf_pq_codebook_len
        );
    }
}
