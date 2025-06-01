use anyhow::{anyhow, Result};
use byteorder::{ByteOrder, LittleEndian};
use compression::compression::IntSeqDecoder;
use compression::elias_fano::ef::{EliasFanoDecoder, EliasFanoDecodingIterator};
use log::debug;
use ouroboros::self_referencing;
use utils::on_disk_ordered_map::encoder::VarintIntegerCodec;
use utils::on_disk_ordered_map::map::OnDiskOrderedMap;

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
                    &mmap,
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
                length: length,
            });
        }

        let offset = *self.inner.borrow_offsets_offset() as usize + term_id as usize * 8;
        let offset = LittleEndian::read_u64(&self.inner.borrow_mmap()[offset..offset + 8]) as usize;

        let next_offset = *self.inner.borrow_offsets_offset() as usize + (term_id + 1) as usize * 8;
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
    pub fn get_posting_list_iterator(&self, term_id: u64) -> Result<EliasFanoDecodingIterator> {
        let offset_len = self.get_term_offset_len(term_id);
        if offset_len.is_none() {
            return Err(anyhow!("Term ID {} is out of bound", term_id));
        }
        let offset_len = offset_len.unwrap();
        let offset = offset_len.offset;
        let length = offset_len.length;

        debug!(
            "[get_posting_list_iterator] Offset: {}, Length: {}",
            offset, length
        );

        let byte_slice = &self.inner.borrow_mmap()[offset..offset + length];
        let decoder = EliasFanoDecoder::new_decoder(byte_slice)
            .expect("Failed to create posting list decoder");
        Ok(decoder.get_iterator(byte_slice))
    }
}

#[cfg(test)]
mod tests {
    use tempdir::TempDir;

    use super::TermIndex;
    use crate::terms::builder::TermBuilder;
    use crate::terms::writer::TermWriter;

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
        let base_directory = temp_dir.path().to_str().unwrap().to_string();

        let mut builder = TermBuilder::new(&base_directory);
        builder.add(0, "a".to_string()).unwrap();
        builder.add(0, "c".to_string()).unwrap();
        builder.add(1, "b".to_string()).unwrap();
        builder.add(2, "c".to_string()).unwrap();

        builder.build().unwrap();
        let writer = TermWriter::new(base_directory.clone());
        writer.write(&mut builder).unwrap();

        let path = format!("{}/combined", base_directory);
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
        assert_eq!(it.next().is_none(), true);

        let mut it = index.get_posting_list_iterator(1).unwrap();
        assert_eq!(it.next().unwrap(), 0);
        assert_eq!(it.next().unwrap(), 2);
        assert_eq!(it.next().is_none(), true);

        let mut it = index.get_posting_list_iterator(2).unwrap();
        assert_eq!(it.next().unwrap(), 1);
        assert_eq!(it.next().is_none(), true);
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
        let base_directory = temp_dir.path().to_str().unwrap().to_string();

        let mut builder = TermBuilder::new(&base_directory);

        // Create 20 docs, each with 5 terms. Some terms are shared.
        let common_terms = vec!["apple", "banana", "orange", "grape", "kiwi"];
        for doc_id in 0..20 {
            for _ in 0..5 {
                let term_idx = doc_id % common_terms.len(); // Cycle through common terms
                let term = common_terms[term_idx].to_string();
                builder.add(doc_id as u64, term).unwrap();
            }
            // Add a unique term for each doc to ensure some distinctness
            builder
                .add(doc_id as u64, format!("unique_term_{}", doc_id))
                .unwrap();
        }

        builder.build().unwrap();
        let writer = TermWriter::new(base_directory.clone());
        writer.write(&mut builder).unwrap();

        let path = format!("{}/combined", base_directory);
        let file_len = std::fs::metadata(&path).unwrap().len();
        let index = TermIndex::new(path, 0, file_len as usize).unwrap();

        // Verify some shared terms
        let apple_id = index.get_term_id("apple").unwrap();
        let mut it = index.get_posting_list_iterator(apple_id).unwrap();
        let mut apple_docs: Vec<u64> = Vec::new();
        while let Some(doc) = it.next() {
            apple_docs.push(doc);
        }
        // "apple" should appear in docs 0, 5, 10, 15
        assert_eq!(apple_docs, vec![0, 5, 10, 15]);

        let banana_id = index.get_term_id("banana").unwrap();
        let mut it = index.get_posting_list_iterator(banana_id).unwrap();
        let mut banana_docs: Vec<u64> = Vec::new();
        while let Some(doc) = it.next() {
            banana_docs.push(doc);
        }
        // "banana" should appear in docs 0, 1, 5, 6, 10, 11, 15, 16
        assert_eq!(banana_docs, vec![1, 6, 11, 16]);

        // Verify a unique term
        let unique_term_5_id = index.get_term_id("unique_term_5").unwrap();
        let mut it = index.get_posting_list_iterator(unique_term_5_id).unwrap();
        assert_eq!(it.next().unwrap(), 5);
        assert_eq!(it.next().is_none(), true);

        // Verify a term that appears in many documents
        let orange_id = index.get_term_id("orange").unwrap();
        let mut it = index.get_posting_list_iterator(orange_id).unwrap();
        let mut orange_docs: Vec<u64> = Vec::new();
        while let Some(doc) = it.next() {
            orange_docs.push(doc);
        }
        // "orange" should appear in docs 0, 1, 2, 5, 6, 7, 10, 11, 12, 15, 16, 17
        assert_eq!(orange_docs, vec![2, 7, 12, 17]);

        let unique_5 = index.get_term_id("unique_term_5").unwrap();
        let mut it = index.get_posting_list_iterator(unique_5).unwrap();
        assert_eq!(it.next().unwrap(), 5);
        assert_eq!(it.next().is_none(), true);
    }
}
