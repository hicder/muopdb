use std::fs::File;
use std::io::{BufWriter, Write};

use anyhow::{anyhow, Result};
use bitvec::prelude::*;
use utils::io::wrap_write;
use utils::mem::transmute_u8_to_slice;

use crate::compression::{CompressionInt, IntSeqDecoder, IntSeqEncoder};

pub struct EliasFano<T: CompressionInt = u64> {
    #[cfg(any(debug_assertions, test))]
    universe: T,
    num_elem: usize,
    lower_bits: BitVec<u64>,
    upper_bits: BitVec<u64>,
    lower_bit_mask: T,
    lower_bit_length: usize,
    // Needed for multiple calls to `encode()`
    cur_high: T,
    cur_index: usize,
}

// TODO(tyb): consider moving this to utils
fn msb<T: CompressionInt>(n: T) -> u64 {
    if n == T::zero() {
        0
    } else {
        (T::bits() - 1) as u64 - n.leading_zeros() as u64
    }
}

impl<T: CompressionInt> EliasFano<T> {
    /// Creates a new EliasFano structure
    pub fn new(universe: T, num_elem: usize) -> Self {
        // lower_bit_length = floor(log(universe / num_elem))
        // More efficient way to do it is with bit manipulation
        let lower_bit_length = if universe > T::from_u64(num_elem as u64) {
            let ratio = universe / T::from_u64(num_elem as u64);
            msb(ratio) as usize
        } else {
            0
        };

        let lower_bit_mask = if lower_bit_length == 0 {
            T::zero()
        } else {
            (T::one() << lower_bit_length) - T::one()
        };

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
            cur_high: T::zero(),
            cur_index: 0,
        }
    }

    /// Returns the value at the given index
    #[allow(dead_code)]
    fn get(&self, index: usize) -> Result<T> {
        if index >= self.num_elem {
            return Err(anyhow!("Index {} out of bound", index));
        }

        // Calculate the position in upper bits
        let mut high = T::zero();
        let mut pos = 0;

        // Calculate the high part of the value
        for _ in 0..index + 1 {
            while pos < self.upper_bits.len() && !self.upper_bits[pos] {
                // Add the gap to high
                high += T::one();
                pos += 1;
            }
            // Skip the '1' that terminates the unary code
            pos += 1;
        }

        // Calculate the low part of the value
        let mut low = T::zero();
        if self.lower_bit_length > 0 {
            let low_start = index * self.lower_bit_length;
            let mut remaining_bits = self.lower_bit_length;
            let mut offset = 0;

            while remaining_bits > 0 {
                let bits_to_load = std::cmp::min(remaining_bits, 64);
                let chunk_start = low_start + offset;
                let chunk = self.lower_bits[chunk_start..chunk_start + bits_to_load].load::<u64>();
                low = low | (T::from_u64(chunk) << offset);
                offset += bits_to_load;
                remaining_bits -= bits_to_load;
            }
        }

        Ok((high << self.lower_bit_length) | low)
    }
}

impl<T: CompressionInt> IntSeqEncoder<T> for EliasFano<T> {
    fn new_encoder(universe: T, num_elem: usize) -> Self {
        Self::new(universe, num_elem)
    }

    fn encode_batch(&mut self, slice: &[T]) -> Result<()> {
        for val in slice {
            self.encode_value(val)?;
        }
        Ok(())
    }

    // Algorithm described in https://vigna.di.unimi.it/ftp/papers/QuasiSuccinctIndices.pdf
    fn encode_value(&mut self, value: &T) -> Result<()> {
        let val = *value;
        // Sanity check only in debug or test builds
        #[cfg(any(debug_assertions, test))]
        if val > self.universe {
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

            // Store lower bits in chunks of 64 bits
            let mut remaining_bits = self.lower_bit_length;
            let mut offset = 0;

            while remaining_bits > 0 {
                let bits_to_store = std::cmp::min(remaining_bits, 64);
                let chunk = (low >> offset).as_u64();
                let chunk_start = start + offset;
                self.lower_bits[chunk_start..chunk_start + bits_to_store].store(chunk);
                offset += bits_to_store;
                remaining_bits -= bits_to_store;
            }
        }

        // Encode upper bits using unary coding
        let high = val >> self.lower_bit_length;
        // Sanity check only in debug or test builds
        #[cfg(any(debug_assertions, test))]
        if high < self.cur_high {
            return Err(anyhow!("Sequence is not sorted"));
        }

        let gap = (high - self.cur_high).as_u64();
        if gap > usize::MAX as u64 {
            return Err(anyhow!(
                "Gap {} too large for Elias-Fano encoding on this platform (max: {})",
                gap,
                usize::MAX
            ));
        }

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

pub struct EliasFanoDecoder<T: CompressionInt = u64> {
    num_elem: usize,
    lower_vec_len: usize,
    upper_vec_len: usize,
    lower_bit_length: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: CompressionInt> EliasFanoDecoder<T> {
    const METADATA_SIZE: usize = 4;
}

impl<T: CompressionInt> IntSeqDecoder<T> for EliasFanoDecoder<T> {
    type IteratorType<'a>
        = EliasFanoDecodingIterator<'a, T>
    where
        T: 'a;

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
            _phantom: std::marker::PhantomData,
        })
    }

    fn get_iterator<'a>(&self, byte_slice: &'a [u8]) -> Self::IteratorType<'a>
    where
        T: 'a,
    {
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
            cumulative_gap_sum: T::zero(),

            lower_bits_slice,
            upper_bits_slice,
            lower_bit_length: self.lower_bit_length,
        }
    }
}

pub struct EliasFanoDecodingIterator<'a, T: CompressionInt = u64> {
    num_elem: usize,
    cur_elem_index: usize,
    cur_upper_bit_index: usize,
    cumulative_gap_sum: T,

    lower_bits_slice: &'a BitSlice<u64>,
    upper_bits_slice: &'a BitSlice<u64>,
    lower_bit_length: usize,
}

impl<'a, T: CompressionInt> EliasFanoDecodingIterator<'a, T> {
    fn decode_upper_part(&mut self) {
        while self.cur_upper_bit_index < self.upper_bits_slice.len()
            && !self.upper_bits_slice[self.cur_upper_bit_index]
        {
            // Add the gap to cumulative sum
            self.cumulative_gap_sum += T::one();
            self.cur_upper_bit_index += 1;
        }
        // Skip the '1' that terminates the unary code
        self.cur_upper_bit_index += 1;
    }

    fn get_lower_part(&self) -> T {
        let mut low = T::zero();
        if self.lower_bit_length > 0 {
            let offset = self.cur_elem_index * self.lower_bit_length;
            let mut remaining_bits = self.lower_bit_length;
            let mut bit_offset = 0;

            while remaining_bits > 0 {
                let bits_to_load = std::cmp::min(remaining_bits, 64);
                let chunk_start = offset + bit_offset;
                let chunk =
                    self.lower_bits_slice[chunk_start..chunk_start + bits_to_load].load::<u64>();
                low = low | (T::from_u64(chunk) << bit_offset);
                bit_offset += bits_to_load;
                remaining_bits -= bits_to_load;
            }
        }
        low
    }
}

impl<'a, T: CompressionInt> Iterator for EliasFanoDecodingIterator<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cur_elem_index < self.num_elem {
            self.decode_upper_part();
            let upper = self.cumulative_gap_sum;
            let lower = self.get_lower_part();
            self.cur_elem_index += 1;

            Some((upper << self.lower_bit_length) | lower)
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
        let values = vec![5u64, 8, 8, 15, 32];
        let upper_bound = 36u64;
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
        let values = vec![5u64, 8, 7, 15, 32];
        let upper_bound = 36u64;
        let mut ef = EliasFano::new_encoder(upper_bound, values.len());
        assert!(ef.encode_batch(&values).is_err());

        // Test sequence with element exceeding upper bound
        let values = vec![5u64, 8, 8, 15, 32];
        let upper_bound = 31u64;
        let mut ef = EliasFano::new_encoder(upper_bound, values.len());
        assert!(ef.encode_batch(&values).is_err());
    }

    #[test]
    fn test_elias_fano_encoding_u32() {
        let values = vec![5u32, 8, 8, 15, 32];
        let upper_bound = 36u32;
        let mut ef = EliasFano::<u32>::new_encoder(upper_bound, values.len());
        assert!(ef.encode_batch(&values).is_ok());

        for i in 0..values.len() {
            let decoded_value = ef.get(i).expect("Failed to decode value");
            assert_eq!(values[i], decoded_value);
        }
    }

    #[test]
    fn test_elias_fano_encoding_u128() {
        let values = vec![5u128, 8, 8, 15, 32];
        let upper_bound = 36u128;
        let mut ef = EliasFano::<u128>::new_encoder(upper_bound, values.len());
        assert!(ef.encode_batch(&values).is_ok());

        for i in 0..values.len() {
            let decoded_value = ef.get(i).expect("Failed to decode value");
            assert_eq!(values[i], decoded_value);
        }
    }

    #[test]
    fn test_elias_fano_decoding() {
        let test_cases = vec![
            (vec![5u64, 8, 8, 15, 32], 36),                // Basic case
            (vec![0u64, 1, 2, 3, 4], 5),                   // Start with 0
            (vec![10u64], 20),                             // Single element
            (vec![1000u64, 2000, 3000, 4000, 5000], 6000), // Large numbers
            (vec![2u64, 4, 6, 8, 10], 10),                 // Non-consecutive integers
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
        let upper_bound = 9999u64;

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
        let ef = EliasFano::<u64> {
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
            (vec![5u64, 8, 8, 15, 32], 36),                // Basic case
            (vec![0u64, 1, 2, 3, 4], 5),                   // Start with 0
            (vec![10u64], 20),                             // Single element
            (vec![1000u64, 2000, 3000, 4000, 5000], 6000), // Large numbers
            (vec![2u64, 4, 6, 8, 10], 10),                 // Non-consecutive integers
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

    #[test]
    fn test_elias_fano_decoding_iterator_u32() {
        let values = vec![5u32, 8, 8, 15, 32];
        let upper_bound = 36u32;
        let mut ef = EliasFano::<u32>::new_encoder(upper_bound, values.len());
        assert!(ef.encode_batch(&values).is_ok());

        let temp_dir = TempDir::new("test_elias_fano_decoding_iterator_u32")
            .expect("Failed to create temporary directory");
        let file_path = temp_dir.path().join("test_file");
        let mut file = File::create(&file_path).expect("Failed to create test file");
        let mut writer = BufWriter::new(&mut file);

        assert!(ef.write(&mut writer).is_ok());
        drop(writer);

        let mut file = File::open(&file_path).expect("Failed to open file for read");
        let mut byte_slice = Vec::new();
        assert!(file.read_to_end(&mut byte_slice).is_ok());

        let decoder =
            EliasFanoDecoder::<u32>::new_decoder(&byte_slice).expect("Failed to create decoder");
        let mut iterator = decoder.get_iterator(&byte_slice);
        let mut i = 0;
        while let Some(decoded_value) = iterator.next() {
            assert_eq!(values[i], decoded_value);
            i += 1;
        }

        let _ = remove_dir_all(&file_path);
    }

    #[test]
    fn test_elias_fano_decoding_iterator_u128() {
        let values = vec![5u128, 8, 8, 15, 32];
        let upper_bound = 36u128;
        let mut ef = EliasFano::<u128>::new_encoder(upper_bound, values.len());
        assert!(ef.encode_batch(&values).is_ok());

        let temp_dir = TempDir::new("test_elias_fano_decoding_iterator_u128")
            .expect("Failed to create temporary directory");
        let file_path = temp_dir.path().join("test_file");
        let mut file = File::create(&file_path).expect("Failed to create test file");
        let mut writer = BufWriter::new(&mut file);

        assert!(ef.write(&mut writer).is_ok());
        drop(writer);

        let mut file = File::open(&file_path).expect("Failed to open file for read");
        let mut byte_slice = Vec::new();
        assert!(file.read_to_end(&mut byte_slice).is_ok());

        let decoder =
            EliasFanoDecoder::<u128>::new_decoder(&byte_slice).expect("Failed to create decoder");
        let mut iterator = decoder.get_iterator(&byte_slice);
        let mut i = 0;
        while let Some(decoded_value) = iterator.next() {
            assert_eq!(values[i], decoded_value);
            i += 1;
        }

        let _ = remove_dir_all(&file_path);
    }

    #[test]
    fn test_elias_fano_u128_near_u64_max() {
        let values = vec![
            u64::MAX as u128 - 1,
            u64::MAX as u128,
            u64::MAX as u128 + 1,
            u64::MAX as u128 * 2,
        ];
        let upper_bound = u64::MAX as u128 * 2 + 1000;
        let mut ef = EliasFano::<u128>::new_encoder(upper_bound, values.len());
        assert!(ef.encode_batch(&values).is_ok());

        for (i, &expected) in values.iter().enumerate() {
            let decoded = ef.get(i).expect("Failed to decode");
            assert_eq!(expected, decoded, "Failed at index {}", i);
        }
    }

    #[test]
    fn test_elias_fano_u128_large_values() {
        let values = vec![
            1u128 << 65,  // Requires 2 chunks (65 bits)
            1u128 << 95,  // Requires 2 chunks (95 bits)
            1u128 << 127, // Requires 2 chunks (127 bits)
            u128::MAX,    // Requires 2 chunks (128 bits)
        ];
        let upper_bound = u128::MAX;
        let mut ef = EliasFano::<u128>::new_encoder(upper_bound, values.len());
        assert!(ef.encode_batch(&values).is_ok());

        for (i, &expected) in values.iter().enumerate() {
            let decoded = ef.get(i).expect("Failed to decode");
            assert_eq!(expected, decoded, "Failed at index {}", i);
        }
    }

    #[test]
    fn test_elias_fano_chunked_storage() {
        // Test different bit lengths that require different numbers of chunks
        let test_cases = vec![
            (vec![100u128], 200u128, 7), // 7 bits - fits in one chunk
            (
                vec![100000000000000000000u128],
                200000000000000000000u128,
                67,
            ), // 65 bits - requires two chunks
            (vec![u128::MAX], u128::MAX, 127), // 127 bits - requires two chunks
        ];

        for (values, upper_bound, expected_bit_length) in test_cases {
            let mut ef = EliasFano::<u128>::new_encoder(upper_bound, values.len());
            assert!(ef.encode_batch(&values).is_ok());

            assert_eq!(
                ef.lower_bit_length, expected_bit_length,
                "Wrong bit length for upper_bound {}",
                upper_bound
            );

            for (i, &expected) in values.iter().enumerate() {
                let decoded = ef.get(i).expect("Failed to decode");
                assert_eq!(
                    expected, decoded,
                    "Failed for upper_bound {} at index {}",
                    upper_bound, i
                );
            }
        }
    }

    #[test]
    fn test_elias_fano_u128_gap_calculation() {
        let values = vec![
            1000u128,
            10000000000000000000u128, // Large gap
            u128::MAX - 1000,
            u128::MAX,
        ];
        let upper_bound = u128::MAX;
        let mut ef = EliasFano::<u128>::new_encoder(upper_bound, values.len());

        assert!(ef.encode_batch(&values).is_ok());

        for (i, &expected) in values.iter().enumerate() {
            let decoded = ef.get(i).expect("Failed to decode");
            assert_eq!(expected, decoded);
        }
    }

    #[test]
    fn test_elias_fano_u128_usize_safety() {
        let large_value = u64::MAX as u128 + 1000; // Too large for usize on 32-bit

        let values = vec![large_value];
        let upper_bound = large_value + 1000;
        let mut ef = EliasFano::<u128>::new_encoder(upper_bound, values.len());

        assert!(ef.encode_batch(&values).is_ok());

        let decoded = ef.get(0).expect("Failed to decode");
        assert_eq!(large_value, decoded);
    }

    #[test]
    fn test_elias_fano_u128_round_trip() {
        let values = vec![
            1u128,
            1000u128,
            u64::MAX as u128,
            u64::MAX as u128 + 1000,
            1u128 << 100,
            u128::MAX - 1,
            u128::MAX,
        ];
        let upper_bound = u128::MAX;

        // Encode
        let mut encoder = EliasFano::<u128>::new_encoder(upper_bound, values.len());
        assert!(encoder.encode_batch(&values).is_ok());

        // Write to temporary file
        let temp_dir = TempDir::new("test_elias_fano_u128_round_trip")
            .expect("Failed to create temporary directory");
        let file_path = temp_dir.path().join("test_file");
        let mut file = File::create(&file_path).expect("Failed to create test file");
        let mut writer = BufWriter::new(&mut file);
        assert!(encoder.write(&mut writer).is_ok());
        drop(writer);

        // Read back
        let mut file = File::open(&file_path).expect("Failed to open file for reading");
        let mut byte_data = Vec::new();
        file.read_to_end(&mut byte_data)
            .expect("Failed to read file");

        // Decode
        let decoder =
            EliasFanoDecoder::<u128>::new_decoder(&byte_data).expect("Failed to create decoder");
        let mut iterator = decoder.get_iterator(&byte_data);

        for (i, expected) in values.iter().enumerate() {
            let decoded = iterator
                .next()
                .expect(&format!("Missing value at index {}", i));
            assert_eq!(*expected, decoded, "Mismatch at index {}", i);
        }

        assert!(iterator.next().is_none(), "Extra values in iterator");
    }

    #[test]
    fn test_elias_fano_u128_value_too_large() {
        let values = vec![1000u128];
        let upper_bound = 500u128; // Too small
        let mut ef = EliasFano::<u128>::new_encoder(upper_bound, values.len());

        let result = ef.encode_batch(&values);
        assert!(result.is_err(), "Should have failed with value > universe");
    }

    #[test]
    fn test_elias_fano_u128_large_sequence() {
        let n = 1000;
        let values: Vec<u128> = (0..n).map(|i| (i as u128) * 10000000000000000000).collect();
        let upper_bound = values.last().unwrap() + 1000;

        let mut ef = EliasFano::<u128>::new_encoder(upper_bound, values.len());
        let encode_result = ef.encode_batch(&values);
        assert!(encode_result.is_ok(), "Encoding failed for large sequence");

        // Verify all values can be decoded
        for (i, &expected) in values.iter().enumerate() {
            let decoded = ef.get(i).expect(&format!("Failed to decode index {}", i));
            assert_eq!(expected, decoded, "Mismatch at index {}", i);
        }
    }
}
