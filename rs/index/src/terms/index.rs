use std::sync::Arc;

use anyhow::{anyhow, Result};
use byteorder::{ByteOrder, LittleEndian};
use compression::compression::IntSeqDecoder;
use compression::elias_fano::ef::{EliasFanoDecoder, EliasFanoDecodingIterator};
use dashmap::DashMap;
use log::debug;
use ouroboros::self_referencing;
use utils::on_disk_ordered_map::encoder::VarintIntegerCodec;
use utils::on_disk_ordered_map::map::OnDiskOrderedMap;

use crate::terms::term_index_info::TermIndexInfoHashTable;

pub struct TermIndex {
    // Box the inner data to make it self-referential
    inner: TermIndexInner,
}

#[allow(dead_code)]
pub struct OffsetLength {
    offset: usize,
    length: usize,
}

#[self_referencing]
struct TermIndexInner {
    mmap: memmap2::Mmap,

    #[borrows(mmap)]
    #[covariant]
    term_map: OnDiskOrderedMap<'this, VarintIntegerCodec>,

    offsets_offset: usize,
    pl_offset: usize,
    offset_len: u64,
}

impl TermIndex {
    pub fn new(path: String, start: usize, len: usize) -> Result<Self> {
        // Make sure start is 8 bytes aligned
        if start % 8 != 0 {
            return Err(anyhow::anyhow!(
                "Start offset {} must be 8 bytes aligned",
                start
            ));
        }

        let backing_file = std::fs::File::open(&path)?;
        let mmap = unsafe {
            memmap2::MmapOptions::new()
                .offset(start as u64)
                .len(len)
                .map(&backing_file)?
        };

        // read first 8 bytes, and store into term_map_len
        let term_map_len = LittleEndian::read_u64(&mmap[0..8]);
        let offset_len = LittleEndian::read_u64(&mmap[8..16]);

        let term_map_offset = 24;
        let offsets_offset = term_map_offset + term_map_len as usize;
        let offsets_offset = offsets_offset.div_ceil(8) * 8;

        let pl_offset = offsets_offset + offset_len as usize;
        let inner = TermIndexInnerBuilder {
            term_map_builder: |mmap| {
                OnDiskOrderedMap::<VarintIntegerCodec>::new(
                    path,
                    mmap,
                    term_map_offset,
                    term_map_offset + term_map_len as usize,
                )
                .unwrap()
            },
            mmap,
            offsets_offset,
            pl_offset,
            offset_len,
        }
        .build();

        Ok(TermIndex { inner })
    }

    pub fn get_term_id(&self, term: &str) -> Option<u64> {
        self.inner.borrow_term_map().get(term)
    }

    /// Retrieves the offset and length of the posting list for a given term ID.
    ///
    /// This function calculates the byte offset and length within the underlying memory map
    /// where the posting list for the specified `term_id` is stored. It handles the special
    /// case for the last term ID to correctly determine its length.
    ///
    /// # Arguments
    ///
    /// * `term_id` - The unique identifier of the term.
    ///
    /// # Returns
    ///
    /// An `Option` containing an `OffsetLength` struct with the calculated offset and length
    /// if the `term_id` is valid, otherwise `None`.
    pub fn get_term_offset_len(&self, term_id: u64) -> Option<OffsetLength> {
        if term_id == self.inner.borrow_offset_len() / 8 - 1 {
            let offset = self.inner.borrow_offsets_offset() + term_id as usize * 8;
            let offset =
                LittleEndian::read_u64(&self.inner.borrow_mmap()[offset..offset + 8]) as usize;
            let length = self.inner.borrow_mmap().len() - (offset + self.inner.borrow_pl_offset());
            return Some(OffsetLength {
                offset: offset + self.inner.borrow_pl_offset(),
                length,
            });
        }

        let offset = *self.inner.borrow_offsets_offset() + term_id as usize * 8;
        let offset = LittleEndian::read_u64(&self.inner.borrow_mmap()[offset..offset + 8]) as usize;

        let next_offset = *self.inner.borrow_offsets_offset() + (term_id + 1) as usize * 8;
        let next_offset =
            LittleEndian::read_u64(&self.inner.borrow_mmap()[next_offset..next_offset + 8])
                as usize;
        Some(OffsetLength {
            offset: offset + self.inner.borrow_pl_offset(),
            length: next_offset - offset,
        })
    }

    /// Returns an iterator over the posting list for a given term ID.
    ///
    /// This function retrieves the offset and length of the posting list associated with the
    /// provided `term_id` from the term index. It then creates an `EliasFanoDecoder` to
    /// decompress the posting list data and returns an iterator that can be used to
    /// traverse the document IDs in the posting list.
    ///
    /// # Arguments
    ///
    /// * `term_id` - The unique identifier of the term for which to retrieve the posting list.
    ///
    /// # Returns
    ///
    /// A `Result` containing an `EliasFanoDecodingIterator` on success, or an `anyhow::Error`
    /// if the `term_id` is out of bounds or if there's an issue creating the decoder.
    pub fn get_posting_list_iterator(
        &self,
        term_id: u64,
    ) -> Result<EliasFanoDecodingIterator<u32>> {
        let offset_len = self.get_term_offset_len(term_id);
        let offset_len = match offset_len {
            Some(ol) => ol,
            None => {
                return Err(anyhow!("Term ID {} is out of bound", term_id));
            }
        };
        let offset = offset_len.offset;
        let length = offset_len.length;

        debug!("[get_posting_list_iterator] Offset: {offset}, Length: {length}");

        let byte_slice = &self.inner.borrow_mmap()[offset..offset + length];
        let decoder = EliasFanoDecoder::new_decoder(byte_slice)
            .expect("Failed to create posting list decoder");
        Ok(decoder.get_iterator(byte_slice))
    }

    /// Get posting list iterator that doesn't borrow from self
    /// Returns a boxed iterator that owns the data
    pub fn get_posting_list_iterator_owned(
        &self,
        term_id: u64,
    ) -> Result<Box<dyn Iterator<Item = u32> + Send + Sync>> {
        let ef_iter = self.get_posting_list_iterator(term_id)?;
        let results: Vec<u32> = ef_iter.collect();
        Ok(Box::new(results.into_iter()))
    }
}

pub struct MultiTermIndex {
    /// Map of user ID to their [`TermIndex`]
    term_indexes: DashMap<u128, Arc<TermIndex>>,
    /// Hash table holding offsets and lengths for all users
    user_index_info: TermIndexInfoHashTable,
    /// Base directory for the term indexes
    base_directory: String,
}

impl MultiTermIndex {
    /// Load the MultiTermIndex and the TermIndexInfo hash table
    pub fn new(base_directory: String) -> Result<Self> {
        // Load the user index info hash table
        let info_file_path = format!("{}/user_term_index_info", base_directory);
        let user_index_info = TermIndexInfoHashTable::load(info_file_path)?;

        Ok(Self {
            base_directory,
            term_indexes: DashMap::new(),
            user_index_info,
        })
    }

    /// Lazily load or return cached TermIndex for a user
    /// Errors: if user not found or TermIndex creation fails
    pub fn get_or_create_index(&self, user_id: u128) -> Result<Arc<TermIndex>> {
        // Try to get from cache first
        if let Some(term_index) = self.term_indexes.get(&user_id) {
            return Ok(term_index.clone());
        }

        // Not in cache, create new one
        let index_info = self
            .user_index_info
            .hash_table
            .get(&user_id)
            .ok_or_else(|| anyhow!("User not found"))?;

        let combined_path = format!("{}/combined", self.base_directory);
        let term_index = TermIndex::new(
            combined_path,
            index_info.offset as usize,
            index_info.length as usize,
        )
        .map_err(|e| anyhow!("Failed to create TermIndex: {e}"))?;

        let term_index_arc = Arc::new(term_index);
        self.term_indexes.insert(user_id, term_index_arc.clone());
        Ok(term_index_arc)
    }

    /// Retrieve the term ID for a given user and term string
    /// Will create/load the TermIndex for the user if not already cached
    /// Errors: if user or term not found or TermIndex creation fails
    pub fn get_term_id_for_user(&self, user_id: u128, term: &str) -> Result<u64> {
        let term_index = self.get_or_create_index(user_id)?;
        term_index
            .get_term_id(term)
            .ok_or_else(|| anyhow!("Term not found"))
    }

    /// Remove and return the TermIndex for a given user
    /// Will create/load the TermIndex for the user if not already cached
    /// Errors: if user not found or TermIndex creation fails
    pub fn take_index_for_user(&self, user_id: u128) -> Result<Arc<TermIndex>> {
        self.get_or_create_index(user_id)?;
        self.term_indexes
            .remove(&user_id)
            .map(|(_, term_index)| term_index)
            .ok_or_else(|| anyhow!("User not found"))
    }

    #[cfg(test)]
    pub fn term_index_info(&self) -> &TermIndexInfoHashTable {
        &self.user_index_info
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;
    use std::sync::Arc;

    use tempdir::TempDir;

    use super::TermIndex;
    use crate::terms::builder::{MultiTermBuilder, TermBuilder};
    use crate::terms::index::MultiTermIndex;
    use crate::terms::writer::{MultiTermWriter, TermWriter};

    /// Tests the basic functionality of the `TermIndex` including term ID retrieval,
    /// offset and length calculation for posting lists, and iteration over posting lists.
    ///
    /// This test performs the following steps:
    /// 1. Initializes a `TermBuilder` and adds a few terms with associated document IDs.
    /// 2. Builds the term data and writes it to a combined file using `TermWriter`.
    /// 3. Creates a `TermIndex` instance from the generated file.
    /// 4. Verifies that `get_term_id` correctly retrieves the IDs for the added terms.
    /// 5. Asserts the correctness of `get_term_offset_len` by checking the calculated
    ///    offsets and lengths for the posting lists of specific terms.
    /// 6. Iterates through the posting lists using `get_posting_list_iterator` and
    ///    confirms that the correct document IDs are returned for each term.
    #[test]
    fn test_term_index() {
        let temp_dir = TempDir::new("test_term_index").unwrap();
        let base_directory = temp_dir.path();
        let base_dir_str = base_directory
            .to_str()
            .expect("Base directory should be valid UTF-8")
            .to_string();
        let mut builder = TermBuilder::new(base_directory.join("scratch.tmp").as_path()).unwrap();
        builder.add(0, "a".to_string()).unwrap();
        builder.add(0, "c".to_string()).unwrap();
        builder.add(1, "b".to_string()).unwrap();
        builder.add(2, "c".to_string()).unwrap();

        builder.build().unwrap();
        let writer = TermWriter::new(base_dir_str.clone());
        writer.write(&mut builder).unwrap();

        let path = format!("{base_dir_str}/combined");
        let file_len = std::fs::metadata(&path).unwrap().len();
        let index = TermIndex::new(path, 0, file_len as usize).unwrap();

        assert_eq!(index.get_term_id("a").unwrap(), 0);
        assert_eq!(index.get_term_id("c").unwrap(), 1);
        assert_eq!(index.get_term_id("b").unwrap(), 2);

        let term_a = index.get_term_offset_len(0).unwrap();
        assert_eq!(term_a.offset, 72);
        assert_eq!(term_a.length, 40);

        let term_c = index.get_term_offset_len(1).unwrap();
        assert_eq!(term_c.offset, 112);
        assert_eq!(term_c.length, 40);

        let term_b = index.get_term_offset_len(2).unwrap();
        assert_eq!(term_b.offset, 152);
        assert_eq!(term_b.length, 40);

        let mut it = index.get_posting_list_iterator(0).unwrap();
        assert_eq!(it.next().unwrap(), 0);
        assert!(it.next().is_none());

        let mut it = index.get_posting_list_iterator(1).unwrap();
        assert_eq!(it.next().unwrap(), 0);
        assert_eq!(it.next().unwrap(), 2);
        assert!(it.next().is_none());

        let mut it = index.get_posting_list_iterator(2).unwrap();
        assert_eq!(it.next().unwrap(), 1);
        assert!(it.next().is_none());
    }

    /// Tests the `TermIndex`'s ability to correctly handle and retrieve posting lists
    /// for terms that are shared across multiple documents, as well as unique terms.
    ///
    /// This test simulates a scenario with 20 documents, where each document contains
    /// a mix of common terms (e.g., "apple", "banana") that appear in multiple documents
    /// and a unique term specific to that document. It verifies that:
    /// 1. The `TermIndex` can be successfully created and loaded from the built data.
    /// 2. `get_term_id` correctly returns the ID for both shared and unique terms.
    /// 3. `get_posting_list_iterator` returns accurate posting lists for shared terms,
    ///    containing all expected document IDs where the term appears.
    /// 4. `get_posting_list_iterator` returns accurate posting lists for unique terms.
    #[test]
    fn test_term_index_with_shared_terms() {
        let temp_dir = TempDir::new("test_term_index_shared").unwrap();
        let base_directory = temp_dir.path();
        let base_dir_str = base_directory.to_str().unwrap().to_string();

        let mut builder = TermBuilder::new(base_directory.join("scratch.tmp").as_path()).unwrap();

        // Create 20 docs, each with 5 terms. Some terms are shared.
        let common_terms = ["apple", "banana", "orange", "grape", "kiwi"];
        for doc_id in 0..20 {
            for _ in 0..5 {
                let term_idx = doc_id % common_terms.len(); // Cycle through common terms
                let term = common_terms[term_idx].to_string();
                builder.add(doc_id as u32, term).unwrap();
            }
            // Add a unique term for each doc to ensure some distinctness
            builder
                .add(doc_id as u32, format!("unique_term_{doc_id}"))
                .unwrap();
        }

        builder.build().unwrap();
        let writer = TermWriter::new(base_dir_str.clone());
        writer.write(&mut builder).unwrap();

        let path = format!("{base_dir_str}/combined");
        let file_len = std::fs::metadata(&path).unwrap().len();
        let index = TermIndex::new(path, 0, file_len as usize).unwrap();

        // Verify some shared terms
        let apple_id = index.get_term_id("apple").unwrap();
        let it = index.get_posting_list_iterator(apple_id).unwrap();
        let mut apple_docs: Vec<u32> = Vec::new();
        for doc in it {
            apple_docs.push(doc);
        }
        // "apple" should appear in docs 0, 5, 10, 15
        assert_eq!(apple_docs, vec![0, 5, 10, 15]);

        let banana_id = index.get_term_id("banana").unwrap();
        let it = index.get_posting_list_iterator(banana_id).unwrap();
        let mut banana_docs: Vec<u32> = Vec::new();
        for doc in it {
            banana_docs.push(doc);
        }
        // "banana" should appear in docs 1, 6, 11, 16
        assert_eq!(banana_docs, vec![1, 6, 11, 16]);

        // Verify a unique term
        let unique_term_5_id = index.get_term_id("unique_term_5").unwrap();
        let mut it = index.get_posting_list_iterator(unique_term_5_id).unwrap();
        assert_eq!(it.next().unwrap(), 5);
        assert!(it.next().is_none());
        // Verify a term that appears in many documents
        let orange_id = index.get_term_id("orange").unwrap();
        let it = index.get_posting_list_iterator(orange_id).unwrap();
        let mut orange_docs: Vec<u32> = Vec::new();
        for doc in it {
            orange_docs.push(doc);
        }
        // "orange" should appear in docs 2, 7, 12, 17
        assert_eq!(orange_docs, vec![2, 7, 12, 17]);

        for doc_id in 0..20 {
            let unique_term = format!("unique_term_{doc_id}");
            let term_id = index.get_term_id(&unique_term).unwrap();
            let mut it = index.get_posting_list_iterator(term_id).unwrap();
            assert_eq!(it.next().unwrap(), doc_id);
            assert!(it.next().is_none());
        }
    }

    fn build_and_write_index(base_dir: &str) -> (Vec<u128>, MultiTermIndex) {
        let mut multi_builder = MultiTermBuilder::new(base_dir.to_string());
        let user1 = 1001u128;
        let user2 = 2002u128;
        let user3 = 3003u128;

        // User 1 terms
        multi_builder
            .add(user1, 10, "apple:red".to_string())
            .unwrap();
        multi_builder
            .add(user1, 20, "banana:yellow".to_string())
            .unwrap();
        multi_builder
            .add(user1, 30, "apple:green".to_string())
            .unwrap();

        // User 2 terms
        multi_builder
            .add(user2, 11, "car:toyota".to_string())
            .unwrap();
        multi_builder
            .add(user2, 22, "car:honda".to_string())
            .unwrap();
        multi_builder
            .add(user2, 33, "bike:yamaha".to_string())
            .unwrap();

        // User 3 single term
        multi_builder
            .add(user3, 1, "test:value".to_string())
            .unwrap();

        multi_builder.build().unwrap();
        let writer = MultiTermWriter::new(base_dir.to_string());
        writer.write(&mut multi_builder).unwrap();

        let index = MultiTermIndex::new(base_dir.to_string()).unwrap();
        (vec![user1, user2, user3], index)
    }

    #[test]
    fn test_multi_term_index_roundtrip() {
        let tmp = TempDir::new("multi_term_index_roundtrip").unwrap();
        let base_dir = tmp.path().to_str().unwrap().to_string();

        let (users, index) = build_and_write_index(&base_dir);
        let user1 = users[0];
        let user2 = users[1];

        // Term IDs should differ within the same user
        let id_apple_red = index.get_term_id_for_user(user1, "apple:red").unwrap();
        let id_banana = index.get_term_id_for_user(user1, "banana:yellow").unwrap();
        assert_ne!(id_apple_red, id_banana);

        // Verify isolation of lookup between users
        assert!(
            index
                .get_term_id_for_user(user1, "car:toyota")
                .unwrap_err()
                .to_string()
                == "Term not found",
            "User1 should not see User2’s terms"
        );
        assert!(
            index
                .get_term_id_for_user(user2, "apple:red")
                .unwrap_err()
                .to_string()
                == "Term not found",
            "User2 should not see User1’s terms"
        );

        // Check that we can open posting list iterator and it matches builder data
        let term_index_apple = index.get_or_create_index(user1).unwrap();
        let pl_apple: Vec<_> = term_index_apple
            .get_posting_list_iterator(id_apple_red)
            .unwrap()
            .collect();
        assert_eq!(pl_apple, vec![10]);

        let term_id = index.get_term_id_for_user(user2, "bike:yamaha").unwrap();
        let term_index_bike = index.get_or_create_index(user2).unwrap();
        let pl_bike: Vec<_> = term_index_bike
            .get_posting_list_iterator(term_id)
            .unwrap()
            .collect();
        assert_eq!(pl_bike, vec![33]);
    }

    #[test]
    fn test_multi_term_index_empty() {
        let tmp = TempDir::new("multi_term_index_empty").unwrap();
        let base_dir = tmp.path().to_str().unwrap().to_string();

        let mut builder = MultiTermBuilder::new(base_dir.clone());
        builder.build().unwrap();

        let writer = MultiTermWriter::new(base_dir.clone());
        writer.write(&mut builder).unwrap();

        let combined_path = format!("{}/combined", base_dir);
        assert!(Path::new(&combined_path).exists());
        let len = fs::metadata(&combined_path).unwrap().len();
        assert_eq!(len, 0, "Combined file should be empty");

        let index_info_path = format!("{}/user_term_index_info", base_dir);
        assert!(Path::new(&index_info_path).exists());

        let index = MultiTermIndex::new(base_dir.clone()).unwrap();
        // The hash table should be empty
        assert_eq!(index.term_index_info().hash_table.len(), 0);
        // Attempting to get a user should fail
        assert!(index.get_or_create_index(123u128).is_err());
    }

    #[test]
    fn test_multi_term_index_alignment_and_ranges() {
        let tmp = TempDir::new("multi_term_index_alignment").unwrap();
        let base_dir = tmp.path().to_str().unwrap().to_string();

        let (users, index) = build_and_write_index(&base_dir);
        let info = index.term_index_info();

        // Combined file size
        let combined_len = fs::metadata(format!("{}/combined", base_dir))
            .unwrap()
            .len() as u64;

        // Verify alignment and non-overlap
        let mut last_end = 0;
        let mut offsets = vec![];

        for user_id in &users {
            let entry = info.hash_table.get(user_id).unwrap();
            assert_eq!(entry.offset % 8, 0, "Offset not 8-byte aligned");
            assert!(entry.offset + entry.length <= combined_len);
            assert!(entry.offset >= last_end, "Overlapping regions");
            last_end = entry.offset + entry.length;
            offsets.push(entry.offset);
        }

        // Ensure deterministic order (sorted offsets)
        let mut sorted_offsets = offsets.clone();
        sorted_offsets.sort();
        assert_eq!(offsets, sorted_offsets);
    }

    #[test]
    fn test_multi_term_index_lazy_load_cache() {
        let tmp = TempDir::new("multi_term_index_lazy_cache").unwrap();
        let base_dir = tmp.path().to_str().unwrap().to_string();

        let (users, index) = build_and_write_index(&base_dir);
        let user1 = users[0];

        // First call should load
        let first = Arc::as_ptr(&index.get_or_create_index(user1).unwrap());
        // Second call should reuse
        let second = Arc::as_ptr(&index.get_or_create_index(user1).unwrap());
        assert_eq!(first, second, "Should return cached TermIndex reference");

        // Should fail for unknown user
        let unknown_user = 99999u128;
        assert!(index.get_or_create_index(unknown_user).is_err());
    }
}
