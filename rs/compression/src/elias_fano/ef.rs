use std::fs::File;
use std::io::{BufWriter, Write};

use anyhow::{anyhow, Result};
use bitvec::prelude::*;
use utils::io::wrap_write;
use utils::mem::transmute_u8_to_slice;

use crate::compression::{IntSeqDecoder, IntSeqEncoder};

pub struct EliasFano {
    #[cfg(any(debug_assertions, test))]
    universe: usize,
    num_elem: usize,
    lower_bits: BitVec<u64>,
    upper_bits: BitVec<u64>,
    lower_bit_mask: u64,
    lower_bit_length: usize,
    // Needed for multiple calls to `encode()`
    cur_high: u64,
    cur_index: usize,
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
    pub fn new(universe: usize, num_elem: usize) -> Self {
        // lower_bit_length = floor(log(universe / num_elem))
        // More efficient way to do it is with bit manipulation
        let lower_bit_length = if universe > num_elem {
            msb((universe / num_elem) as u64)
        } else {
            0
        } as usize;
        let lower_bit_mask = (1 << lower_bit_length) - 1;
        let mut lower_bits = BitVec::with_capacity(num_elem * lower_bit_length);
        // Ensure lower_bits is filled with false initially
        lower_bits.resize(num_elem * lower_bit_length, false);

        // The upper bits are encoded using unary coding for the gaps between consecutive values.
        // This part uses at most 2n bits:
        // - There are exactly n '1' bits, one for each of the n elements in the sequence.
        // - The number of '0' bits is at most n, representing the gaps between the high bits of
        // consecutive elements (the total number of possible distinct values that can be
        // represented by the high parts is limited by the number of elements in the sequence)
        Self {
            #[cfg(any(debug_assertions, test))]
            universe,
            num_elem,
            lower_bits,
            upper_bits: BitVec::with_capacity(2 * num_elem),
            lower_bit_mask,
            lower_bit_length,
            cur_high: 0,
            cur_index: 0,
        }
    }

    /// Returns the value at the given index
    #[allow(dead_code)]
    fn get(&self, index: usize) -> Result<u64> {
        if index >= self.num_elem {
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
    fn new_encoder(universe: usize, num_elem: usize) -> Self {
        Self::new(universe, num_elem)
    }

    // Algorithm described in https://vigna.di.unimi.it/ftp/papers/QuasiSuccinctIndices.pdf
    fn encode_batch(&mut self, slice: &[u64]) -> Result<()> {
        for &val in slice.iter() {
            self.encode_value(&val)?;
        }
        Ok(())
    }

    fn encode_value(&mut self, value: &u64) -> Result<()> {
        let val = *value;
        // Sanity check only in debug or test builds
        #[cfg(any(debug_assertions, test))]
        if val > self.universe as u64 {
            return Err(anyhow!(
                "Element {}th ({}) is greater than universe",
                self.cur_index,
                val
            ));
        }
        // Encode lower bits efficiently
        if self.lower_bit_length > 0 {
            let low = val & self.lower_bit_mask;
            let start = self.cur_index * self.lower_bit_length;
            self.lower_bits[start..start + self.lower_bit_length].store(low as u64);
        }

        // Encode upper bits using unary coding
        let high = val >> self.lower_bit_length;
        // Sanity check only in debug or test builds
        #[cfg(any(debug_assertions, test))]
        if high < self.cur_high {
            return Err(anyhow!("Sequence is not sorted"));
        }

        let gap = high - self.cur_high;
        self.upper_bits
            .extend_from_bitslice(&BitVec::<u8>::repeat(false, gap as usize));
        self.upper_bits.push(true);

        self.cur_high = high;
        self.cur_index += 1;
        Ok(())
    }

    fn len(&self) -> usize {
        let lower_vec: &[u64] = self.lower_bits.as_raw_slice();
        let upper_vec: &[u64] = self.upper_bits.as_raw_slice();
        (1 /* self.num_elem */ +
         1 /* self.lower_bit_length */ +
         1 /* lower_vec.len() */ +
         1 /* upper_vec.len() */ +
         lower_vec.len() +
         upper_vec.len())
            * std::mem::size_of::<u64>()
    }

    fn write(&self, writer: &mut BufWriter<&mut File>) -> Result<usize> {
        let mut total_bytes_written = wrap_write(writer, &((self.num_elem as u64).to_le_bytes()))?;
        total_bytes_written += wrap_write(writer, &((self.lower_bit_length as u64).to_le_bytes()))?;
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

pub struct EliasFanoDecoder {
    num_elem: usize,
    lower_vec_len: usize,
    upper_vec_len: usize,
    lower_bit_length: usize,
}

impl EliasFanoDecoder {
    const METADATA_SIZE: usize = 4;
}

impl IntSeqDecoder for EliasFanoDecoder {
    type IteratorType<'a> = EliasFanoDecodingIterator<'a>;
    type Item = u64;

    fn new_decoder(byte_slice: &[u8]) -> Result<Self> {
        let encoded_data = transmute_u8_to_slice::<u64>(byte_slice);
        if encoded_data.len() < Self::METADATA_SIZE {
            return Err(anyhow!("Not enough metadata for EliasFano encoded data"));
        }
        let [num_elem, lower_bit_length, lower_vec_len, upper_vec_len, ..] =
            encoded_data[..Self::METADATA_SIZE]
        else {
            return Err(anyhow!("Invalid metadata for EliasFano encoded data"));
        };

        Ok(Self {
            num_elem: num_elem as usize,
            lower_vec_len: lower_vec_len as usize,
            upper_vec_len: upper_vec_len as usize,
            lower_bit_length: lower_bit_length as usize,
        })
    }

    fn get_iterator<'a>(&self, byte_slice: &'a [u8]) -> Self::IteratorType<'a> {
        let encoded_data = transmute_u8_to_slice::<u64>(byte_slice);
        let lower_bits_start = Self::METADATA_SIZE;
        let upper_bits_start = Self::METADATA_SIZE + self.lower_vec_len;
        let lower_bits_slice =
            BitSlice::<u64>::from_slice(&encoded_data[lower_bits_start..upper_bits_start]);
        let upper_bits_slice = BitSlice::<u64>::from_slice(
            &encoded_data[upper_bits_start..upper_bits_start + self.upper_vec_len],
        );
        EliasFanoDecodingIterator {
            num_elem: self.num_elem,
            cur_elem_index: 0,
            cur_upper_bit_index: 0,
            cumulative_gap_sum: 0,

            lower_bits_slice,
            upper_bits_slice,
            lower_bit_mask: (1 << self.lower_bit_length) - 1,
            lower_bit_length: self.lower_bit_length,
        }
    }
}

pub struct EliasFanoDecodingIterator<'a> {
    num_elem: usize,
    cur_elem_index: usize,
    cur_upper_bit_index: usize,
    cumulative_gap_sum: usize,

    lower_bits_slice: &'a BitSlice<u64>,
    upper_bits_slice: &'a BitSlice<u64>,
    lower_bit_mask: u64,
    lower_bit_length: usize,
}

impl<'a> EliasFanoDecodingIterator<'a> {
    fn decode_upper_part(&mut self) {
        while self.cur_upper_bit_index < self.upper_bits_slice.len()
            && !self.upper_bits_slice[self.cur_upper_bit_index]
        {
            // Add the gap to cumulative sum
            self.cumulative_gap_sum += 1;
            self.cur_upper_bit_index += 1;
        }
        // Skip the '1' that terminates the unary code
        self.cur_upper_bit_index += 1;
    }

    fn get_lower_part(&self) -> usize {
        let mut low = 0;
        if self.lower_bit_length > 0 {
            let offset = self.cur_elem_index * self.lower_bit_length;
            low = (self.lower_bits_slice[offset..offset + self.lower_bit_length].load::<u64>()
                & self.lower_bit_mask) as usize;
        }
        low
    }
}

impl<'a> Iterator for EliasFanoDecodingIterator<'a> {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cur_elem_index < self.num_elem {
            self.decode_upper_part();
            let upper = self.cumulative_gap_sum;
            let lower = self.get_lower_part();
            self.cur_elem_index += 1;

            Some((upper << self.lower_bit_length | lower) as u64)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs::{remove_dir_all, File};
    use std::io::{BufReader, BufWriter, Read};

    use tempdir::TempDir;

    use super::*;

    #[test]
    fn test_elias_fano_encoding() {
        let values = vec![5, 8, 8, 15, 32];
        let upper_bound = 36;
        let mut ef = EliasFano::new_encoder(upper_bound, values.len());
        assert!(ef.encode_batch(&values).is_ok());

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
        assert!(ef.encode_batch(&values).is_err());

        // Test sequence with element exceeding upper bound
        let values = vec![5, 8, 8, 15, 32];
        let upper_bound = 31;
        let mut ef = EliasFano::new_encoder(upper_bound, values.len());
        assert!(ef.encode_batch(&values).is_err());
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
            assert!(ef.encode_batch(&values).is_ok());

            for i in 0..values.len() {
                let decoded_value = ef.get(i).expect("Failed to decode value");
                assert_eq!(values[i], decoded_value);
            }
        }

        // Test random access on a larger set
        let values: Vec<u64> = (1..=100).collect(); // Sorted list from 1 to 100
        let upper_bound = 9999;

        let mut ef = EliasFano::new_encoder(upper_bound, values.len());
        assert!(ef.encode_batch(&values).is_ok());

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
            num_elem: 5,
            lower_bits: BitVec::from_slice(&[0b10101010_01010101]),
            upper_bits: BitVec::from_slice(&[0b11001100_00110011]),
            lower_bit_mask: 0b1111,
            lower_bit_length: 4,
            cur_high: 0,
            cur_index: 0,
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
            5, 0, 0, 0, 0, 0, 0, 0, // num_elem (5 as u64)
            4, 0, 0, 0, 0, 0, 0, 0, // lower_bit_length (4 as u64)
            1, 0, 0, 0, 0, 0, 0, 0, // lower_vec.len() (1 as u64)
            1, 0, 0, 0, 0, 0, 0, 0, // upper_vec.len() (1 as u64)
            0b01010101, 0b10101010, 0, 0, 0, 0, 0, 0, // lower_bits
            0b00110011, 0b11001100, 0, 0, 0, 0, 0, 0, // upper_bits
        ];

        assert_eq!(written_data, expected_data);
        assert_eq!(bytes_written, expected_data.len());
    }

    #[test]
    fn test_elias_fano_decoding_iterator() {
        let test_cases = vec![
            (vec![5, 8, 8, 15, 32], 36),                // Basic case
            (vec![0, 1, 2, 3, 4], 5),                   // Start with 0
            (vec![10], 20),                             // Single element
            (vec![1000, 2000, 3000, 4000, 5000], 6000), // Large numbers
            (vec![2, 4, 6, 8, 10], 10),                 // Non-consecutive integers
        ];

        for (values, upper_bound) in test_cases {
            let mut ef = EliasFano::new_encoder(upper_bound, values.len());
            assert!(ef.encode_batch(&values).is_ok());

            let temp_dir = TempDir::new("test_elias_fano_decoding_iterator")
                .expect("Failed to create temporary directory");
            let file_path = temp_dir.path().join("test_file");
            let mut file = File::create(&file_path).expect("Failed to create test file");
            let mut writer = BufWriter::new(&mut file);

            // Call the write method
            assert!(ef.write(&mut writer).is_ok());

            drop(writer);

            // Read the file contents into a byte vector
            let mut file = File::open(&file_path).expect("Failed to open file for read");
            let mut byte_slice = Vec::new();
            assert!(file.read_to_end(&mut byte_slice).is_ok());

            let decoder = EliasFanoDecoder::new_decoder(&byte_slice)
                .expect("Failed to create posting list decoder");
            let mut i = 0;
            for idx in decoder.get_iterator(&byte_slice) {
                assert_eq!(values[i], idx);
                i += 1;
            }

            let _ = remove_dir_all(&file_path);
        }
    }
}
