use std::fs::File;
use std::io::{BufWriter, Write};

use anyhow::{anyhow, Result};
use compression::compression::IntSeqEncoder;
use compression::elias_fano::ef::EliasFano;
use log::debug;
use utils::io::{append_file_to_writer, wrap_write};
use utils::on_disk_ordered_map::encoder::{IntegerCodec, VarintIntegerCodec};

use crate::terms::builder::TermBuilder;

pub struct TermWriter {
    base_dir: String,
}

impl TermWriter {
    pub fn new(base_dir: String) -> Self {
        Self { base_dir }
    }

    /// Writes the term map, posting lists, and their offsets to disk.
    ///
    /// The data is organized into three main components:
    /// 1.  **Term Map**: A mapping from term IDs to term strings, stored as an `OnDiskOrderedMap`.
    /// 2.  **Posting Lists**: Elias-Fano encoded lists of document IDs for each term.
    /// 3.  **Offsets**: A list of offsets indicating the start position of each posting list within the posting lists file.
    ///
    /// These components are written into separate temporary files (`term_map`, `posting_lists`, `offsets`)
    /// and then combined into a single `combined` file in the following format:
    ///
    /// ```text
    /// +---------------------+
    /// | term_map_len (8B)   |
    /// +---------------------+
    /// | offsets_len (8B)    |
    /// +---------------------+
    /// | posting_lists_len (8B)|
    /// +---------------------+
    /// | Term Map Data       |
    /// | (padded to 8-byte   |
    /// |  alignment)         |
    /// +---------------------+
    /// | Offsets Data        |
    /// +---------------------+
    /// | Posting Lists Data  |
    /// +---------------------+
    /// ```
    ///
    /// # Arguments
    /// * `builder` - A mutable reference to a `TermBuilder` instance that has been built.
    ///
    /// # Returns
    /// `Result<()>` indicating success or an error if the `TermBuilder` is not built or an I/O error occurs.
    pub fn write(&self, builder: &mut TermBuilder) -> Result<()> {
        if !builder.is_built() {
            return Err(anyhow!("TermBuilder is not built"));
        }

        if builder.num_terms() == 0 {
            return Ok(());
        }

        // Write the term map
        let term_map_path = format!("{}/term_map", self.base_dir);
        builder
            .term_map
            .build(VarintIntegerCodec::new(), term_map_path.as_str())?;

        // Get length of term_map file
        let term_map_file = File::open(term_map_path.as_str()).unwrap();
        let term_map_len = term_map_file.metadata().unwrap().len();

        // Writer for posting lists
        let posting_list_path = format!("{}/posting_lists", self.base_dir);
        let mut pl_file = File::create(posting_list_path.as_str()).unwrap();
        let mut pl_writer = BufWriter::new(&mut pl_file);

        let num_terms = builder.num_terms();
        let mut offsets = Vec::with_capacity(num_terms as usize);
        let mut last_pl_offset = 0;
        for term_id in 0..num_terms {
            // elias fano encode
            let posting_list = builder.get_posting_list_by_id(term_id).unwrap();
            let mut encoder =
                EliasFano::new_encoder(*posting_list.last().unwrap(), posting_list.len());
            encoder.encode_batch(posting_list).unwrap();
            let len = encoder.write(&mut pl_writer)?;
            debug!(
                "[write] Term ID: {}, Offset: {}, Length: {}",
                term_id, last_pl_offset, len
            );
            offsets.push(last_pl_offset);
            last_pl_offset += len;
        }

        pl_writer.flush()?;

        // Write the offsets
        let offset_path = format!("{}/offsets", self.base_dir);
        let mut offset_file = File::create(offset_path.as_str()).unwrap();
        let mut offset_writer = BufWriter::new(&mut offset_file);
        let mut offset_len = 0;
        for offset in offsets {
            offset_len += wrap_write(&mut offset_writer, &offset.to_le_bytes())?;
        }
        offset_writer.flush()?;

        #[cfg(debug_assertions)]
        {
            // Open the file
            let pl_file = File::open(posting_list_path.as_str()).unwrap();
            let pl_file_len = pl_file.metadata().unwrap().len();
            assert_eq!(pl_file_len, last_pl_offset as u64);

            let offset_file = File::open(offset_path.as_str()).unwrap();
            let offset_file_len = offset_file.metadata().unwrap().len();
            assert_eq!(offset_file_len, offset_len as u64);
        }

        // Print the length for each file, in the same line
        debug!(
            "Term map length: {}, Offset length: {}, Posting list length: {}",
            term_map_len, offset_len, last_pl_offset
        );

        // Write the combined file. First, write the length of term_map file, then length of offsets file, then length of posting lists file
        let combined_path = format!("{}/combined", self.base_dir);
        let mut combined_file = File::create(combined_path.as_str()).unwrap();
        let mut combined_writer = BufWriter::new(&mut combined_file);
        wrap_write(&mut combined_writer, &term_map_len.to_le_bytes())?;
        wrap_write(&mut combined_writer, &offset_len.to_le_bytes())?;
        wrap_write(&mut combined_writer, &last_pl_offset.to_le_bytes())?;

        let mut total_size = 24;
        append_file_to_writer(term_map_path.as_str(), &mut combined_writer)?;
        total_size += term_map_len as usize;

        let padding = 8 - (total_size % 8);
        if padding != 8 {
            let padding_buffer = vec![0; padding];
            total_size += wrap_write(&mut combined_writer, &padding_buffer)?;
        }

        append_file_to_writer(offset_path.as_str(), &mut combined_writer)?;
        total_size += offset_len;
        let padding = 8 - (total_size % 8);
        if padding != 8 {
            let padding_buffer = vec![0; padding];
            total_size += wrap_write(&mut combined_writer, &padding_buffer)?;
        }

        append_file_to_writer(posting_list_path.as_str(), &mut combined_writer)?;
        total_size += last_pl_offset;
        combined_writer.flush()?;

        debug!("Total size: {}", total_size);

        // Remove term_map, offsets, posting lists files
        // std::fs::remove_file(term_map_path.as_str())?;
        // std::fs::remove_file(offset_path.as_str())?;
        // std::fs::remove_file(posting_list_path.as_str())?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_term_writer() {
        let tmp_dir = tempdir::TempDir::new("test_term_writer").unwrap();
        let base_directory = tmp_dir.path().to_str().unwrap();

        let mut builder = TermBuilder::new(base_directory);
        for i in 0..10 {
            builder.add(i, format!("key{}", i % 3)).unwrap();
        }

        builder.build().unwrap();

        let writer = TermWriter::new(base_directory.to_string());
        writer.write(&mut builder).unwrap();

        // Check the files
        let combined_path = format!("{}/combined", base_directory);
        let combined_file = File::open(combined_path.as_str()).unwrap();
        let combined_file_len = combined_file.metadata().unwrap().len();
        assert_eq!(combined_file_len, 216);
    }
}
