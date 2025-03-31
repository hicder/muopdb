use std::collections::BTreeMap;

use anyhow::Result;

use super::encoder::IntegerCodec;

pub struct OnDiskOrderedMap<C: IntegerCodec> {
    path: String,
    mmap: memmap2::Mmap,
    codec: C,

    index: BTreeMap<String, u64>,
    data_offset: usize,
}

impl<C: IntegerCodec> OnDiskOrderedMap<C> {
    pub fn new(path: String) -> Result<Self> {
        let mmap = unsafe { memmap2::Mmap::map(&std::fs::File::open(&path).unwrap()) }.unwrap();
        let codec = C::new();
        let mut map = BTreeMap::new();

        if mmap[0] != codec.id() {
            return Err(anyhow::anyhow!("Codec id mismatch"));
        }

        // TODO: read the index
        let mut offset = 1;
        let index_len = u64::from_le_bytes(mmap[offset..offset + 8].try_into().unwrap());
        offset += 8;

        let mut read_bytes = 0 as usize;
        while read_bytes < index_len as usize {
            let res = codec.decode_u32(&mmap[offset..]);
            offset += res.num_bytes_read;
            read_bytes += res.num_bytes_read;

            let key_len = res.value as usize;
            let key = &mmap[offset..offset + key_len];
            offset += key_len;
            read_bytes += key_len;

            let res = codec.decode_u64(&mmap[offset..]);
            offset += res.num_bytes_read;
            read_bytes += res.num_bytes_read;

            let value = res.value;
            map.insert(String::from_utf8(key.to_vec()).unwrap(), value);
        }

        println!("index_len: {}", index_len);

        Ok(OnDiskOrderedMap {
            path,
            mmap,
            codec,
            index: map,
            data_offset: offset,
        })
    }

    pub fn get(&self, key: &str) -> Option<u64> {
        let idx = self.index.upper_bound(bounds::Bound::Included(key));
        if idx == self.index.end() {
            return None;
        }

        let (k, v) = idx.next_back().unwrap();
        if k == key {
            return Some(*v);
        }

        None
    }
}
