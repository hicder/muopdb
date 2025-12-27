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
    let byte_count = std::mem::size_of_val(slice);
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, byte_count) }
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

pub fn ids_to_u128s(ids: &[Id]) -> Result<Vec<u128>, String> {
    ids.iter().map(id_to_u128).collect()
}

/// Converts a UUID string (with or without hyphens) to u128.
/// Supports formats like "550e8400-e29b-41d4-a716-446655440000" or "550e8400e29b41d4a716446655440000".
pub fn uuid_str_to_u128(uuid: &str) -> Result<u128, String> {
    let hex_str: String = uuid.chars().filter(|c| *c != '-').collect();
    if hex_str.len() != 32 {
        return Err(format!(
            "Invalid UUID length: expected 32 hex chars, got {}",
            hex_str.len()
        ));
    }
    u128::from_str_radix(&hex_str, 16).map_err(|e| format!("Invalid UUID hex: {}", e))
}

/// Converts a single Id proto to u128.
/// Returns an error if neither uuid nor (low_id AND high_id) are set.
pub fn id_to_u128(id: &Id) -> Result<u128, String> {
    if let Some(uuid) = &id.uuid {
        uuid_str_to_u128(uuid)
    } else {
        match (id.low_id, id.high_id) {
            (Some(low), Some(high)) => Ok(low as u128 | (high as u128) << 64),
            _ => Err("Either uuid or both low_id and high_id must be set".to_string()),
        }
    }
}

/// Converts a u128 to a hyphenated UUID string.
pub fn u128_to_uuid_str(id: u128) -> String {
    let s = format!("{:032x}", id);
    format!(
        "{}-{}-{}-{}-{}",
        &s[0..8],
        &s[8..12],
        &s[12..16],
        &s[16..20],
        &s[20..32]
    )
}

/// Converts a u128 to an Id proto with all fields populated.
pub fn u128_to_id(id: u128) -> Id {
    Id {
        low_id: Some(id as u64),
        high_id: Some((id >> 64) as u64),
        uuid: Some(u128_to_uuid_str(id)),
    }
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
        let lows = [
            0x123456789abcdef0,
            0x123456789abcdef1,
            0x123456789abcdef2,
            0x123456789abcdef3,
            0x123456789abcdef4,
        ];
        let highs = [0x4312, 0x4312, 0x4312, 0x4312, 0x4312];

        let id_proto: Vec<Id> = lows
            .iter()
            .zip(highs.iter())
            .map(|(&low_id, &high_id)| Id {
                low_id: Some(low_id),
                high_id: Some(high_id),
                uuid: None,
            })
            .collect();

        let ids = ids_to_u128s(&id_proto).unwrap();

        assert_eq!(
            ids,
            [
                0x4312123456789abcdef0,
                0x4312123456789abcdef1,
                0x4312123456789abcdef2,
                0x4312123456789abcdef3,
                0x4312123456789abcdef4
            ]
        );
    }

    #[test]
    fn test_uuid_parsing() {
        let uuid_str = "550e8400-e29b-41d4-a716-446655440000";
        let expected: u128 = 0x550e8400e29b41d4a716446655440000;
        assert_eq!(uuid_str_to_u128(uuid_str).unwrap(), expected);

        let uuid_str_no_hyphen = "550e8400e29b41d4a716446655440000";
        assert_eq!(uuid_str_to_u128(uuid_str_no_hyphen).unwrap(), expected);

        // Invalid hex
        assert!(uuid_str_to_u128("550e8400-e29b-41d4-a716-44665544000G").is_err());
        // Invalid length
        assert!(uuid_str_to_u128("550e8400-e29b-41d4-a716").is_err());
    }

    #[test]
    fn test_id_to_u128_with_uuid() {
        let id = Id {
            low_id: Some(123),
            high_id: Some(456),
            uuid: Some("550e8400-e29b-41d4-a716-446655440000".to_string()),
        };
        // UUID should take precedence
        assert_eq!(id_to_u128(&id).unwrap(), 0x550e8400e29b41d4a716446655440000);
    }

    #[test]
    fn test_id_to_u128_validation() {
        // Missing both
        let id_none = Id {
            low_id: None,
            high_id: None,
            uuid: None,
        };
        assert!(id_to_u128(&id_none).is_err());

        // Missing high
        let id_no_high = Id {
            low_id: Some(123),
            high_id: None,
            uuid: None,
        };
        assert!(id_to_u128(&id_no_high).is_err());

        // Valid low/high
        let id_valid = Id {
            low_id: Some(123),
            high_id: Some(456),
            uuid: None,
        };
        assert_eq!(id_to_u128(&id_valid).unwrap(), 123 | (456 << 64));
    }

    #[test]
    fn test_u128_to_id() {
        let id_val: u128 = 0x550e8400f29b41d4a716446655440000;
        let id_proto = u128_to_id(id_val);

        assert_eq!(id_proto.low_id, Some(0xa716446655440000));
        assert_eq!(id_proto.high_id, Some(0x550e8400f29b41d4));
        assert_eq!(
            id_proto.uuid,
            Some("550e8400-f29b-41d4-a716-446655440000".to_string())
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
            [
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
