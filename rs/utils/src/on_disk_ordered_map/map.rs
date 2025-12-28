use std::collections::BTreeMap;
use std::range::Bound::Included;

use anyhow::Result;

use super::encoder::IntegerCodec;

#[allow(unused)]
pub struct OnDiskOrderedMap<'a, C: IntegerCodec> {
    path: String,
    mmap: &'a memmap2::Mmap,
    codec: C,

    index: BTreeMap<String, u64>,

    // This map is only valid between [start_offset, end_offset)
    start_offset: usize,
    end_offset: usize,

    data_offset: usize,
}

/// TODO: Add a bloom filter to quickly check if a key exists
impl<'a, C: IntegerCodec> OnDiskOrderedMap<'a, C> {
    pub fn new(
        path: String,
        mmap: &'a memmap2::Mmap,
        start_offset: usize,
        end_offset: usize,
    ) -> Result<Self> {
        // let mmap = unsafe { memmap2::Mmap::map(&std::fs::File::open(&path).unwrap()) }.unwrap();
        let codec = C::new();
        let mut map = BTreeMap::new();

        if mmap[start_offset] != codec.id() {
            return Err(anyhow::anyhow!("Codec id mismatch"));
        }

        // TODO: read the index
        let mut offset = start_offset + 1;
        let decoding_res = codec.decode_u64(mmap[offset..offset + 8].try_into().unwrap());
        offset += decoding_res.num_bytes_read;
        let index_len = decoding_res.value;

        let mut read_bytes = 0_usize;
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

        Ok(OnDiskOrderedMap {
            path,
            mmap,
            codec,
            index: map,
            start_offset,
            end_offset,
            data_offset: offset,
        })
    }

    fn index_for_key(&self, key: &str) -> Option<&u64> {
        // If key is less than the first key, then return None
        if key < self.index.first_key_value().unwrap().0.as_str() {
            return None;
        }

        let mut cursor = self.index.upper_bound(Included(key));
        if let Some((k, v)) = cursor.peek_prev() {
            if k == key {
                return Some(v);
            }
        }

        if let Some((_, v)) = cursor.prev() {
            return Some(v);
        }
        None
    }

    #[allow(unused)]
    pub fn get(&self, key: &str) -> Option<u64> {
        match self.index_for_key(key) {
            Some(offset) => {
                let mut offset = self.data_offset + *offset as usize;
                // TODO: probably have some way to determine max key length.
                let mut prev_key = Vec::<u8>::with_capacity(10);
                loop {
                    let res = self.codec.decode_u32(&self.mmap[offset..]);
                    let shared = res.value;
                    offset += res.num_bytes_read;

                    let res = self.codec.decode_u32(&self.mmap[offset..]);
                    let unshared = res.value;
                    offset += res.num_bytes_read;

                    // copy unshared bytes from mmap[offset..offset+shared] to prev_key, starting from prev_key[shared]
                    prev_key.resize(shared as usize, 0);
                    prev_key.extend_from_slice(&self.mmap[offset..offset + unshared as usize]);
                    offset += unshared as usize;

                    let res = self.codec.decode_u64(&self.mmap[offset..]);
                    offset += res.num_bytes_read;

                    // Check whether we should stop iterating
                    if prev_key.as_slice() == key.as_bytes() {
                        return Some(res.value);
                    } else if prev_key.as_slice() > key.as_bytes() {
                        // println!("Key not found");
                        return None;
                    } else if offset >= self.end_offset {
                        return None;
                    }
                }
            }
            None => None,
        }
    }

    pub fn iter(&self) -> OnDiskOrderedMapIterator<'_, C> {
        OnDiskOrderedMapIterator {
            map: self,
            offset: self.data_offset,
            prev_key: Vec::new(),
        }
    }
}

pub struct OnDiskOrderedMapIterator<'a, C: IntegerCodec> {
    map: &'a OnDiskOrderedMap<'a, C>,
    offset: usize,
    prev_key: Vec<u8>,
}

impl<'a, C: IntegerCodec> Iterator for OnDiskOrderedMapIterator<'a, C> {
    type Item = (String, u64);

    fn next(&mut self) -> Option<Self::Item> {
        if self.offset >= self.map.end_offset {
            return None;
        }

        let res = self.map.codec.decode_u32(&self.map.mmap[self.offset..]);
        let shared = res.value as usize;
        self.offset += res.num_bytes_read;

        let res = self.map.codec.decode_u32(&self.map.mmap[self.offset..]);
        let unshared = res.value as usize;
        self.offset += res.num_bytes_read;

        // Reset prev_key if it was cleared (at block boundaries)
        if shared == 0 {
            self.prev_key.clear();
        } else {
            self.prev_key.truncate(shared);
        }

        self.prev_key
            .extend_from_slice(&self.map.mmap[self.offset..self.offset + unshared]);
        self.offset += unshared;

        let res = self.map.codec.decode_u64(&self.map.mmap[self.offset..]);
        self.offset += res.num_bytes_read;

        let key = String::from_utf8(self.prev_key.clone()).unwrap();
        let value = res.value;

        // If we crossed a block boundary in the next iteration, shared_len will be 0.
        // But we don't know it yet.
        // Actually, the builder clears prev_key at block boundaries.
        // Line 183: prev_key.clear();
        // So shared_len will be 0 at the start of each block.

        // Wait, if shared_len is 0, it means it's the start of a block (or just a key with no shared prefix).
        // The builder does `prev_key.clear()` at block boundary (line 183).
        // This is correct.

        Some((key, value))
    }
}

#[cfg(test)]
mod tests {
    use std::fs::OpenOptions;
    use std::io::Write;

    use rand::Rng;

    use super::*;
    use crate::on_disk_ordered_map::builder::OnDiskOrderedMapBuilder;
    use crate::on_disk_ordered_map::encoder::{FixedIntegerCodec, VarintIntegerCodec};

    #[test]
    fn test_map_varint() {
        let tmp_dir = tempdir::TempDir::new("test_map").unwrap();
        let base_directory = tmp_dir.path().to_str().unwrap();
        let final_map_file_path = base_directory.to_string() + "/map.bin";

        let mut builder = OnDiskOrderedMapBuilder::new();

        builder.add(String::from("key1"), 1);
        builder.add(String::from("key2"), 2);
        builder.add(String::from("key3"), 3);

        let codec = VarintIntegerCodec {};
        builder.build(codec, &final_map_file_path).unwrap();

        let mmap =
            unsafe { memmap2::Mmap::map(&std::fs::File::open(&final_map_file_path).unwrap()) }
                .unwrap();

        let map =
            OnDiskOrderedMap::<VarintIntegerCodec>::new(final_map_file_path, &mmap, 0, mmap.len())
                .unwrap();
        assert_eq!(map.index.len(), 1);
        assert_eq!(map.index.get("key1").unwrap(), &0);
        assert_eq!(map.data_offset, 8);

        assert_eq!(map.get("key1").unwrap(), 1);
        assert_eq!(map.get("key2").unwrap(), 2);
        assert_eq!(map.get("key3").unwrap(), 3);
        assert_eq!(map.get("key0"), None);
        assert_eq!(map.get("key4"), None);
    }

    #[test]
    fn test_map_varint_custom_entries() {
        let tmp_dir = tempdir::TempDir::new("test_map").unwrap();
        let base_directory = tmp_dir.path().to_str().unwrap();
        let final_map_file_path = base_directory.to_string() + "/map.bin";

        let mut builder = OnDiskOrderedMapBuilder::new();

        // Add entries in the order specified: {"a", 0}, {"c", 1}, {"b", 2}
        builder.add(String::from("a"), 0);
        builder.add(String::from("c"), 1);
        builder.add(String::from("b"), 2);

        let codec = VarintIntegerCodec {};
        builder.build(codec, &final_map_file_path).unwrap();

        let mmap =
            unsafe { memmap2::Mmap::map(&std::fs::File::open(&final_map_file_path).unwrap()) }
                .unwrap();

        println!("Mmap len: {}", mmap.len());

        // Read the original content
        let original_content = std::fs::read(&final_map_file_path).unwrap();

        // Generate 24 random bytes
        let mut rng = rand::thread_rng();
        let random_bytes: Vec<u8> = (0..24).map(|_| rng.gen::<u8>()).collect();

        // Prepend random bytes and write back to file
        let mut file = OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(&final_map_file_path)
            .unwrap();
        file.write_all(&random_bytes).unwrap();
        file.write_all(&original_content).unwrap();
        file.flush().unwrap();

        let mmap =
            unsafe { memmap2::Mmap::map(&std::fs::File::open(&final_map_file_path).unwrap()) }
                .unwrap();

        println!("Mmap len: {}", mmap.len());

        let map =
            OnDiskOrderedMap::<VarintIntegerCodec>::new(final_map_file_path, &mmap, 24, mmap.len())
                .unwrap();

        // Assert the index structure
        assert_eq!(map.index.len(), 1);
        assert_eq!(map.index.get("a").unwrap(), &0);

        // Assert that we can retrieve all the entries correctly
        assert_eq!(map.get("a").unwrap(), 0);
        assert_eq!(map.get("b").unwrap(), 2);
        assert_eq!(map.get("c").unwrap(), 1);

        // Assert that non-existent keys return None
        assert_eq!(map.get("d"), None);
        assert_eq!(map.get("z"), None);
        assert_eq!(map.get(""), None);
    }

    #[test]
    fn test_map_fixed() {
        let tmp_dir = tempdir::TempDir::new("test_map").unwrap();
        let base_directory = tmp_dir.path().to_str().unwrap();
        let final_map_file_path = base_directory.to_string() + "/map.bin";

        let mut builder = OnDiskOrderedMapBuilder::new();

        builder.add(String::from("key1"), 1);
        builder.add(String::from("key2"), 2);
        builder.add(String::from("key3"), 3);

        let codec = FixedIntegerCodec {};
        builder.build(codec, &final_map_file_path).unwrap();

        let mmap =
            unsafe { memmap2::Mmap::map(&std::fs::File::open(&final_map_file_path).unwrap()) }
                .unwrap();

        let map =
            OnDiskOrderedMap::<FixedIntegerCodec>::new(final_map_file_path, &mmap, 0, mmap.len())
                .unwrap();
        assert_eq!(map.index.len(), 1);
        assert_eq!(map.index.get("key1").unwrap(), &0);
        assert_eq!(map.data_offset, 25);

        assert_eq!(map.get("key1").unwrap(), 1);
        assert_eq!(map.get("key2").unwrap(), 2);
        assert_eq!(map.get("key3").unwrap(), 3);
        assert_eq!(map.get("key0"), None);
        assert_eq!(map.get("key4"), None);
    }
}
