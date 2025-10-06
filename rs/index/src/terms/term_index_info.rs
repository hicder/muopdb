use std::fs::File;
use std::path::Path;

use memmap2::Mmap;
use odht::{Config, FxHashFn, HashTableOwned};

pub struct TermIndexInfo {
    pub offset: u64,
    pub length: u64,
}

pub struct TermIndexInfoHashTableConfig {}

impl Config for TermIndexInfoHashTableConfig {
    type Key = u128;
    type Value = TermIndexInfo;

    type EncodedKey = [u8; 16];
    type EncodedValue = [u8; 16];

    type H = FxHashFn;

    #[inline]
    fn encode_key(k: &Self::Key) -> Self::EncodedKey {
        k.to_le_bytes()
    }
    #[inline]
    fn encode_value(v: &Self::Value) -> Self::EncodedValue {
        let mut buf = [0u8; 16];
        buf[..8].copy_from_slice(&v.offset.to_le_bytes());
        buf[8..].copy_from_slice(&v.length.to_le_bytes());
        buf
    }
    #[inline]
    fn decode_key(k: &Self::EncodedKey) -> Self::Key {
        u128::from_le_bytes(*k)
    }
    #[inline]
    fn decode_value(v: &Self::EncodedValue) -> Self::Value {
        let offset = u64::from_le_bytes(v[..8].try_into().expect("v should be 16 bytes"));
        let length = u64::from_le_bytes(v[8..].try_into().expect("v should be 16 bytes"));
        TermIndexInfo { offset, length }
    }
}

pub struct TermIndexInfoHashTable {
    #[allow(dead_code)]
    /// Needed to keep the mmap alive while using the hash table
    pub mmap: Mmap,
    /// The actual hash table
    pub hash_table: HashTableOwned<TermIndexInfoHashTableConfig>,
}

impl TermIndexInfoHashTable {
    /// Load from a file path
    pub fn load<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let hash_table = HashTableOwned::<TermIndexInfoHashTableConfig>::from_raw_bytes(&mmap)
            .expect("Failed to create hash table from mmap");
        Ok(Self { mmap, hash_table })
    }
}
