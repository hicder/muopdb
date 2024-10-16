use memmap2::Mmap;
use std::fs::File;

use crate::hnsw_writer::Header;

pub struct Hnsw {
    // Need this for mmap
    #[allow(dead_code)]
    backing_file: File,
    mmap: Mmap,
    header: Header,
    data_offset: usize,
}

impl Hnsw {
    pub fn get_header(&self) -> &Header {
        &self.header
    }

    pub fn get_data_offset(&self) -> usize {
        self.data_offset
    }

    pub fn new(backing_file: File, mmap: Mmap, header: Header, data_offset: usize) -> Self {
        Self {
            backing_file,
            mmap,
            header,
            data_offset,
        }
    }

    pub fn get_edges_slice(&self) -> &[u32] {
        let start = self.data_offset;
        utils::mem::transmute_u8_to_slice(&self.mmap[start..start + self.header.edges_len as usize])
    }

    pub fn get_points_slice(&self) -> &[u32] {
        let start = self.data_offset + self.header.edges_len as usize;
        utils::mem::transmute_u8_to_slice(
            &self.mmap[start..start + self.header.points_len as usize],
        )
    }

    /// Returns the edge offsets slice
    pub fn get_edge_offsets_slice(&self) -> &[u64] {
        let start =
            self.data_offset + self.header.edges_len as usize + self.header.points_len as usize;
        utils::mem::transmute_u8_to_slice(
            &self.mmap[start..start + self.header.edge_offsets_len as usize],
        )
    }

    /// Returns the level offsets slice
    pub fn get_level_offsets_slice(&self) -> &[u64] {
        let start = self.data_offset
            + self.header.edges_len as usize
            + self.header.points_len as usize
            + self.header.edge_offsets_len as usize;
        let slice = &self.mmap[start..start + self.header.level_offsets_len as usize];
        return utils::mem::transmute_u8_to_slice(slice);
    }
}
