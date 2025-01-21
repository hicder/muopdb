use std::fs::OpenOptions;

use anyhow::{anyhow, Result};
use memmap2::MmapMut;

pub struct InvalidatedIdsStorage {
    mmap: MmapMut,
    num_points: usize,
}

const BITS_IN_BYTE: usize = 8;

impl InvalidatedIdsStorage {
    pub fn new(base_directory: &str, num_points: usize) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(format!("{}/invalidated_ids", base_directory))?;

        let bytes_needed = num_points.div_ceil(BITS_IN_BYTE);
        file.set_len(bytes_needed as u64)?;

        let mmap = unsafe { MmapMut::map_mut(&file)? };
        Ok(Self { mmap, num_points })
    }

    pub fn invalidate(&mut self, point_id: u32) -> Result<()> {
        let point_id = point_id as usize;
        if point_id >= self.num_points {
            return Err(anyhow!("ID out of bound"));
        }

        let byte_index = point_id / BITS_IN_BYTE;
        let bit_index = point_id % BITS_IN_BYTE;

        self.mmap[byte_index] |= 1 << bit_index;

        Ok(self.mmap.flush()?)
    }

    // TODO(tyb): consider having an unsafe version w/o bound check
    // for performance reason
    pub fn is_valid(&self, point_id: u32) -> Result<bool> {
        let point_id = point_id as usize;
        if point_id >= self.num_points {
            return Err(anyhow!("ID out of bound"));
        }
        let byte_index = point_id / BITS_IN_BYTE;
        let bit_index = point_id % BITS_IN_BYTE;

        Ok((self.mmap[byte_index] & (1 << bit_index)) == 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invalidated_ids() {
        let temp_dir = tempdir::TempDir::new("test_invalidated_ids")
            .expect("Failed to create temporary directory");
        let base_dir = temp_dir
            .path()
            .to_str()
            .expect("Failed to convert temporary directory path to string")
            .to_string();

        // Test creation and initial state
        let mut bitmap = InvalidatedIdsStorage::new(&base_dir, 1000)
            .expect("Failed to open invalidated ids storage");

        // All ids should be valid initially
        for i in 0..1000 {
            assert!(bitmap.is_valid(i).expect("Failed to query validity state"));
        }

        // Test out of bound
        assert!(bitmap.invalidate(1000).is_err());
        assert!(bitmap.is_valid(1001).is_err());

        // Test invalidating ids
        assert!(bitmap.invalidate(0).is_ok());
        assert!(bitmap.invalidate(500).is_ok());
        assert!(bitmap.invalidate(999).is_ok());

        assert!(!bitmap.is_valid(0).expect("Failed to query validity state"));
        assert!(!bitmap
            .is_valid(500)
            .expect("Failed to query validity state"));
        assert!(!bitmap
            .is_valid(999)
            .expect("Failed to query validity state"));
        assert!(bitmap.is_valid(1).expect("Failed to query validity state"));
        assert!(bitmap
            .is_valid(998)
            .expect("Failed to query validity state"));

        // Test edge cases
        assert!(bitmap.invalidate(7).is_ok());
        assert!(bitmap.invalidate(8).is_ok());
        assert!(!bitmap.is_valid(7).expect("Failed to query validity state"));
        assert!(!bitmap.is_valid(8).expect("Failed to query validity state"));
        assert!(bitmap.is_valid(6).expect("Failed to query validity state"));
        assert!(bitmap.is_valid(9).expect("Failed to query validity state"));

        // Test persistence
        drop(bitmap);
        let bitmap2 = InvalidatedIdsStorage::new(&base_dir, 1000)
            .expect("Failed to open invalidated ids storage");
        assert!(!bitmap2.is_valid(0).expect("Failed to query validity state"));
        assert!(!bitmap2
            .is_valid(500)
            .expect("Failed to query validity state"));
        assert!(!bitmap2
            .is_valid(999)
            .expect("Failed to query validity state"));
        assert!(!bitmap2.is_valid(7).expect("Failed to query validity state"));
        assert!(!bitmap2.is_valid(8).expect("Failed to query validity state"));
    }
}
