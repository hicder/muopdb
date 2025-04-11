use std::ptr;

use proto::muopdb::Id;

pub fn transmute_u8_to_slice<T>(data: &[u8]) -> &[T] {
    unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const T,
            data.len() / std::mem::size_of::<T>(),
        )
    }
}

/// Use when we can guarantee the alignment of data.
pub fn transmute_u8_to_val_aligned<T: Copy>(data: &[u8]) -> T {
    #[cfg(debug_assertions)]
    {
        assert!(data.len() >= std::mem::size_of::<T>());
        assert!(data.as_ptr().align_offset(std::mem::align_of::<T>()) == 0);
    }
    unsafe { *(data.as_ptr() as *const T) }
}

/// Use when we're not sure about the alignment of data
pub fn transmute_u8_to_val_unaligned<T: Copy>(data: &[u8]) -> T {
    #[cfg(debug_assertions)]
    {
        assert!(data.len() >= std::mem::size_of::<T>());
    }
    unsafe { ptr::read_unaligned(data.as_ptr() as *const T) }
}

pub fn transmute_slice_to_u8<T>(slice: &[T]) -> &[u8] {
    let byte_count = slice.len() * std::mem::size_of::<T>();
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, byte_count) }
}

pub fn get_ith_val_from_raw_ptr<T: Copy>(raw_ptr: *const T, index: usize) -> T {
    unsafe { *raw_ptr.add(index) }
}

pub struct LowsAndHighs {
    pub lows: Vec<u64>,
    pub highs: Vec<u64>,
}

pub fn u128s_to_lows_highs(ids: &[u128]) -> LowsAndHighs {
    let mut result = LowsAndHighs {
        lows: Vec::with_capacity(ids.len()),
        highs: Vec::with_capacity(ids.len()),
    };

    ids.iter().for_each(|id| {
        result.lows.push(*id as u64);
        result.highs.push((*id >> 64) as u64);
    });

    result
}

pub fn bytes_to_u128s(bytes: &[u8]) -> Vec<u128> {
    bytes
        .chunks_exact(16)
        .map(|chunk| {
            let mut value: u128 = 0;
            for (i, &byte) in chunk.iter().enumerate() {
                value |= (byte as u128) << (i * 8);
            }
            value
        })
        .collect()
}

pub fn ids_to_u128s(ids: &[Id]) -> Vec<u128> {
    let mut result = Vec::with_capacity(ids.len());

    for id in ids {
        result.push(id.low_id as u128 | (id.high_id as u128) << 64);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test lows and highs
    #[test]
    fn test_lows_and_highs() {
        let ids = vec![
            0x4312123456789abcdef0,
            0x4312123456789abcdef1,
            0x4312123456789abcdef2,
            0x4312123456789abcdef3,
            0x4312123456789abcdef4,
        ];

        let lows_highs = u128s_to_lows_highs(&ids);

        assert_eq!(
            lows_highs.lows,
            vec![
                0x123456789abcdef0,
                0x123456789abcdef1,
                0x123456789abcdef2,
                0x123456789abcdef3,
                0x123456789abcdef4
            ]
        );
        assert_eq!(
            lows_highs.highs,
            vec![0x4312, 0x4312, 0x4312, 0x4312, 0x4312]
        );
    }

    #[test]
    fn test_id_to_u128s() {
        let lows = vec![
            0x123456789abcdef0,
            0x123456789abcdef1,
            0x123456789abcdef2,
            0x123456789abcdef3,
            0x123456789abcdef4,
        ];
        let highs = vec![0x4312, 0x4312, 0x4312, 0x4312, 0x4312];

        let id_proto: Vec<Id> = lows
            .iter()
            .zip(highs.iter())
            .map(|(&low_id, &high_id)| Id { low_id, high_id })
            .collect();

        let ids = ids_to_u128s(&id_proto);

        assert_eq!(
            ids,
            vec![
                0x4312123456789abcdef0,
                0x4312123456789abcdef1,
                0x4312123456789abcdef2,
                0x4312123456789abcdef3,
                0x4312123456789abcdef4
            ]
        );
    }

    #[test]
    fn test_bytes_to_u128s() {
        let mut bytes = Vec::new();

        bytes.extend_from_slice(&[
            0xf0, 0xde, 0xbc, 0x9a, 0x78, 0x56, 0x34, 0x12, 0x12, 0x43, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00,
        ]);

        bytes.extend_from_slice(&[
            0xf1, 0xde, 0xbc, 0x9a, 0x78, 0x56, 0x34, 0x12, 0x12, 0x43, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00,
        ]);

        bytes.extend_from_slice(&[
            0xf2, 0xde, 0xbc, 0x9a, 0x78, 0x56, 0x34, 0x12, 0x12, 0x43, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00,
        ]);

        let ids = bytes_to_u128s(&bytes);

        assert_eq!(
            ids,
            vec![
                0x4312123456789abcdef0,
                0x4312123456789abcdef1,
                0x4312123456789abcdef2,
            ]
        );
    }

    use rkyv::util::AlignedVec;

    /// Write a test for transmute_u8_to_val_aligned
    #[test]
    fn test_transmute_u8_to_val_aligned() {
        let value: u32 = 0x12345678;
        let mut aligned_bytes: AlignedVec<4> = AlignedVec::new();
        aligned_bytes.extend_from_slice(&value.to_le_bytes());
        let transmuted_value: u32 = transmute_u8_to_val_aligned(&aligned_bytes);
        assert_eq!(transmuted_value, value);
    }

    /// Write a test for transmute_u8_to_val_unaligned
    #[test]
    fn test_transmute_u8_to_val_unaligned() {
        let value: u32 = 0x12345678;
        let mut unaligned_bytes: Vec<u8> = Vec::new();
        unaligned_bytes.extend_from_slice(&value.to_le_bytes());
        let transmuted_value: u32 = transmute_u8_to_val_unaligned(&unaligned_bytes);
        assert_eq!(transmuted_value, value);
    }
}
