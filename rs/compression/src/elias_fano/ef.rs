use std::fs::File;
use std::io::{BufWriter, Write};

use anyhow::{anyhow, Result};
use bitvec::prelude::*;
use utils::io::wrap_write;

use crate::compression::IntSeqEncoder;

pub struct EliasFano {
    #[cfg(any(debug_assertions, test))]
    universe: usize,
    size: usize,
    lower_bits: BitVec<u64>,
    upper_bits: BitVec<u64>,
    lower_bit_mask: u64,
    lower_bit_length: usize,
}

// TODO(tyb): consider moving this to utils
fn msb(n: u64) -> u64 {
    if n == 0 {
        0
    } else {
        let highest_index = 63u64;
        highest_index - n.leading_zeros() as u64
    }
}

impl EliasFano {
    /// Creates a new EliasFano structure
    pub fn new(universe: usize, size: usize) -> Self {
        // lower_bit_length = floor(log(universe / size))
        // More efficient way to do it is with bit manipulation
        let lower_bit_length = if universe > size {
            msb((universe / size) as u64)
        } else {
            0
        } as usize;
        let lower_bit_mask = (1 << lower_bit_length) - 1;
        let mut lower_bits = BitVec::with_capacity(size * lower_bit_length);
        // Ensure lower_bits is filled with false initially
        lower_bits.resize(size * lower_bit_length, false);

        // The upper bits are encoded using unary coding for the gaps between consecutive values.
        // This part uses at most 2n bits:
        // - There are exactly n '1' bits, one for each of the n elements in the sequence.
        // - The number of '0' bits is at most n, representing the gaps between the high bits of
        // consecutive elements (the total number of possible distinct values that can be
        // represented by the high parts is limited by the number of elements in the sequence)
        Self {
            #[cfg(any(debug_assertions, test))]
            universe,
            size,
            lower_bits,
            upper_bits: BitVec::with_capacity(2 * size),
            lower_bit_mask,
            lower_bit_length,
        }
    }

    /// Returns the value at the given index
    #[allow(dead_code)]
    fn get(&self, index: usize) -> Result<u64> {
        if index >= self.size {
            return Err(anyhow!("Index {} out of bound", index));
        }

        // Calculate the position in upper bits
        let mut high = 0;
        let mut pos = 0;

        // Calculate the high part of the value
        for _ in 0..index + 1 {
            while pos < self.upper_bits.len() && !self.upper_bits[pos] {
                // Add the gap to high
                high += 1;
                pos += 1;
            }
            // Skip the '1' that terminates the unary code
            pos += 1;
        }

        // Calculate the low part of the value
        let mut low = 0;
        if self.lower_bit_length > 0 {
            let low_start = index * self.lower_bit_length;
            low = (self.lower_bits[low_start..low_start + self.lower_bit_length].load::<u64>()
                & self.lower_bit_mask) as usize;
        }

        Ok((high << self.lower_bit_length | low) as u64)
    }
}

impl IntSeqEncoder for EliasFano {
    fn new_encoder(universe: usize, size: usize) -> Self {
        Self::new(universe, size)
    }

    // Algorithm described in https://vigna.di.unimi.it/ftp/papers/QuasiSuccinctIndices.pdf
    fn encode(&mut self, values: &[u64]) -> Result<()> {
        let mut prev_high = 0;
        for (i, &val) in values.iter().enumerate() {
            // Sanity check only in debug or test builds
            #[cfg(any(debug_assertions, test))]
            if val > self.universe as u64 {
                return Err(anyhow!(
                    "Element {}th ({}) is greater than universe",
                    i,
                    val
                ));
            }
            // Encode lower bits efficiently
            if self.lower_bit_length > 0 {
                let low = val & self.lower_bit_mask;
                let start = i * self.lower_bit_length;
                self.lower_bits[start..start + self.lower_bit_length].store(low as u64);
            }

            // Encode upper bits using unary coding
            let high = val >> self.lower_bit_length;
            // Sanity check only in debug or test builds
            #[cfg(any(debug_assertions, test))]
            if high < prev_high {
                return Err(anyhow!("Sequence is not sorted"));
            }

            let gap = high - prev_high;
            self.upper_bits
                .extend_from_bitslice(&BitVec::<u8>::repeat(false, gap as usize));
            self.upper_bits.push(true);

            prev_high = high;
        }
        Ok(())
    }

    fn len(&self) -> usize {
        self.size
    }

    fn write(&self, writer: &mut BufWriter<&mut File>) -> Result<usize> {
        let mut total_bytes_written =
            wrap_write(writer, &((self.lower_bit_length as u64).to_le_bytes()))?;
        let lower_vec: &[u64] = self.lower_bits.as_raw_slice();
        let upper_vec: &[u64] = self.upper_bits.as_raw_slice();
        total_bytes_written += wrap_write(writer, &((lower_vec.len() as u64).to_le_bytes()))?;
        total_bytes_written += wrap_write(writer, &((upper_vec.len() as u64).to_le_bytes()))?;

        for &val in lower_vec.iter() {
            total_bytes_written += wrap_write(writer, &val.to_le_bytes())?;
        }
        for &val in upper_vec.iter() {
            total_bytes_written += wrap_write(writer, &val.to_le_bytes())?;
        }

        writer.flush()?;

        Ok(total_bytes_written)
    }
}
#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::{BufReader, BufWriter, Read};

    use tempdir::TempDir;

    use super::*;

    #[test]
    fn test_elias_fano_encoding() {
        let values = vec![5, 8, 8, 15, 32];
        let upper_bound = 36;
        let mut ef = EliasFano::new_encoder(upper_bound, values.len());
        assert!(ef.encode(&values).is_ok());

        // Calculate expected lower bits
        // L = floor(log2(36/5)) = 2
        // Lower 2 bits of each value: 01, 00, 00, 11, 00
        // lower_bits: 0011000001
        let expected_lower_bits = bitvec![u64, Lsb0; 1, 0, 0, 0, 0, 0, 1, 1, 0, 0];

        // Calculate expected upper bits
        // Upper bits: 1, 2, 2, 3, 8
        // Gaps: 1, 1, 0, 1, 5
        // Unary encoding: 01|01|1|01|000001
        // upper_bits: 1000001011010
        let expected_upper_bits = bitvec![u64, Lsb0; 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1];

        assert_eq!(ef.lower_bit_length, 2);
        assert_eq!(ef.lower_bits, expected_lower_bits,);
        assert_eq!(ef.upper_bits, expected_upper_bits,);

        // Test unsorted sequence
        let values = vec![5, 8, 7, 15, 32];
        let upper_bound = 36;
        let mut ef = EliasFano::new_encoder(upper_bound, values.len());
        assert!(ef.encode(&values).is_err());

        // Test sequence with element exceeding upper bound
        let values = vec![5, 8, 8, 15, 32];
        let upper_bound = 31;
        let mut ef = EliasFano::new_encoder(upper_bound, values.len());
        assert!(ef.encode(&values).is_err());
    }

    #[test]
    fn test_elias_fano_decoding() {
        let test_cases = vec![
            (vec![5, 8, 8, 15, 32], 36),                // Basic case
            (vec![0, 1, 2, 3, 4], 5),                   // Start with 0
            (vec![10], 20),                             // Single element
            (vec![1000, 2000, 3000, 4000, 5000], 6000), // Large numbers
            (vec![2, 4, 6, 8, 10], 10),                 // Non-consecutive integers
        ];

        for (values, upper_bound) in test_cases {
            let mut ef = EliasFano::new_encoder(upper_bound, values.len());
            assert!(ef.encode(&values).is_ok());

            for i in 0..values.len() {
                let decoded_value = ef.get(i).expect("Failed to decode value");
                assert_eq!(values[i], decoded_value);
            }
        }

        // Test random access on a larger set
        let values: Vec<u64> = (1..=100).collect(); // Sorted list from 1 to 100
        let upper_bound = 9999;

        let mut ef = EliasFano::new_encoder(upper_bound, values.len());
        assert!(ef.encode(&values).is_ok());

        // Check random accesses
        assert_eq!(ef.get(0).expect("Failed to decode value"), 1);
        assert_eq!(ef.get(50).expect("Failed to decode value"), 51);
        assert_eq!(ef.get(99).expect("Failed to decode value"), 100);

        // Test out of bounds
        assert!(ef.get(100).is_err());
    }

    #[test]
    fn test_elias_fano_write() {
        // Create a mock EliasFano instance
        let ef = EliasFano {
            universe: 100,
            size: 5,
            lower_bits: BitVec::from_slice(&[0b10101010_01010101]),
            upper_bits: BitVec::from_slice(&[0b11001100_00110011]),
            lower_bit_mask: 0b1111,
            lower_bit_length: 4,
        };

        let temp_dir =
            TempDir::new("test_elias_fano_write").expect("Failed to create temporary directory");
        let file_path = temp_dir.path().join("test_file");
        let mut file = File::create(&file_path).expect("Failed to create test file");
        let mut writer = BufWriter::new(&mut file);

        // Call the write method
        let bytes_written = ef
            .write(&mut writer)
            .expect("Failed to write encoded sequence");

        // Read the contents of the file
        let mut file = File::open(&file_path).expect("Failed to open test file for reading");
        let mut written_data = Vec::new();
        assert!(BufReader::new(&mut file)
            .read_to_end(&mut written_data)
            .is_ok());

        // Expected data
        let expected_data = vec![
            4, 0, 0, 0, 0, 0, 0, 0, // lower_bit_length (4 as u64)
            1, 0, 0, 0, 0, 0, 0, 0, // lower_vec.len() (1 as u64)
            1, 0, 0, 0, 0, 0, 0, 0, // upper_vec.len() (1 as u64)
            0b01010101, 0b10101010, 0, 0, 0, 0, 0, 0, // lower_bits
            0b00110011, 0b11001100, 0, 0, 0, 0, 0, 0, // upper_bits
        ];

        assert_eq!(written_data, expected_data);
        assert_eq!(bytes_written, expected_data.len());
    }
}
