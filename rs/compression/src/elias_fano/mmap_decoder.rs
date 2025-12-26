use anyhow::{anyhow, Result};
use bitvec::prelude::*;
use utils::mem::transmute_u8_to_slice;

use crate::compression::{CompressionInt, IntSeqDecoder};

pub struct EliasFanoMmapDecoder<T: CompressionInt = u64> {
    num_elem: usize,
    lower_vec_len: usize,
    upper_vec_len: usize,
    lower_bit_length: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: CompressionInt> EliasFanoMmapDecoder<T> {
    const METADATA_SIZE: usize = 4;
}

impl<T: CompressionInt> IntSeqDecoder<T> for EliasFanoMmapDecoder<T> {
    type IteratorType<'a>
        = EliasFanoMMapDecodingIterator<'a, T>
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
        EliasFanoMMapDecodingIterator {
            num_elem: self.num_elem,
            cur_elem_index: 0,
            cur_upper_bit_index: 0,
            cumulative_gap_sum: T::zero(),

            lower_bits_slice,
            upper_bits_slice,
            lower_bit_length: self.lower_bit_length,

            post_skip: false,
        }
    }
}

pub struct EliasFanoMMapDecodingIterator<'a, T: CompressionInt = u64> {
    num_elem: usize,
    cur_elem_index: usize,
    cur_upper_bit_index: usize,
    cumulative_gap_sum: T,

    lower_bits_slice: &'a BitSlice<u64>,
    upper_bits_slice: &'a BitSlice<u64>,
    lower_bit_length: usize,

    // Since `skip_to()` leaves `cur_upper_bit_index` pointing directly at the terminating '1' bit,
    // calling `next()` right after `skip_to()` wouldn't actually process any new gaps, so it'll
    // end up re-decoding and returning the same value before advancing.
    // This indicates to `next()` that a `skip_to()` has just been called previously, so the
    // "current" element has already been peeked at by `skip_to()` and `current()`, and `next()`
    // should advance before decoding.
    post_skip: bool,
}

impl<'a, T: CompressionInt> EliasFanoMMapDecodingIterator<'a, T> {
    /// Skip to the first element >= target
    /// Updates iterator state but doesn't return the value
    /// Call current() or next() to get the value after skip_to
    pub fn skip_to(&mut self, target: T) {
        // Early exit: if we're already exhausted, nothing to do
        if self.cur_elem_index >= self.num_elem {
            return;
        }

        // Early exit: if target is greater than max possible value, mark as exhausted
        // The max value has all upper bits set to max gaps, and all lower bits set to 1
        // For a more precise check, we could store the actual last value, but this is conservative
        let max_possible_high = T::from_u64(self.upper_bits_slice.len() as u64);
        let max_possible_value = (max_possible_high << self.lower_bit_length)
            | ((T::one() << self.lower_bit_length) - T::one());
        if target > max_possible_value {
            self.cur_elem_index = self.num_elem;
            return;
        }

        // Binary search for first element >= target
        let mut left = self.cur_elem_index;
        let mut right = self.num_elem;
        let mut result_index = None;

        while left < right {
            let mid = left + (right - left) / 2;

            if let Some(mid_value) = self.get_value_at_index(mid) {
                if mid_value >= target {
                    result_index = Some(mid);
                    right = mid;
                } else {
                    left = mid + 1;
                }
            } else {
                // Out of bounds, shouldn't happen in valid binary search, but handle gracefully
                break;
            }
        }

        // Position iterator at the found index (or mark as exhausted)
        if let Some(index) = result_index {
            self.seek_to_index(index);
        } else {
            // No element >= target found
            self.cur_elem_index = self.num_elem;
        }
    }

    /// Get the current value without advancing the iterator
    /// Returns None if iterator is exhausted
    pub fn current(&mut self) -> Option<T> {
        self.get_value_at_index(self.cur_elem_index)
    }

    /// Get value at specific index without changing iterator state
    fn get_value_at_index(&self, index: usize) -> Option<T> {
        if index >= self.num_elem {
            return None;
        }

        // Calculate high part by scanning upper bits
        let mut high = T::zero();
        let mut pos = 0;
        let mut elements_seen = 0;

        while pos < self.upper_bits_slice.len() && elements_seen <= index {
            if self.upper_bits_slice[pos] {
                if elements_seen == index {
                    break;
                }
                elements_seen += 1;
            } else {
                high += T::one();
            }
            pos += 1;
        }

        // Get low part
        let low = self.get_lower_part_at_index(index);
        Some((high << self.lower_bit_length) | low)
    }

    /// Seek iterator to specific index (positions before that element)
    fn seek_to_index(&mut self, target_index: usize) {
        if target_index >= self.num_elem {
            self.cur_elem_index = self.num_elem;
            return;
        }

        // Reset state and scan from the beginning to the target index
        self.cur_upper_bit_index = 0;
        self.cumulative_gap_sum = T::zero();
        self.cur_elem_index = 0;

        while self.cur_elem_index < target_index {
            self.decode_upper_part();
            self.cur_elem_index += 1;
        }

        // Now position before target element's '1' (scan to next '1' but don't skip past it)
        while self.cur_upper_bit_index < self.upper_bits_slice.len()
            && !self.upper_bits_slice[self.cur_upper_bit_index]
        {
            self.cumulative_gap_sum += T::one();
            self.cur_upper_bit_index += 1;
        }

        self.post_skip = true;
    }

    /// Get lower part for current element
    fn get_current_lower_part(&self) -> T {
        self.get_lower_part_at_index(self.cur_elem_index)
    }

    /// Get the current position/index in the sequence
    pub fn position(&self) -> usize {
        self.cur_elem_index
    }

    /// Check if iterator is exhausted
    pub fn is_exhausted(&self) -> bool {
        self.cur_elem_index >= self.num_elem
    }

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

    /// Get lower part at specific index
    fn get_lower_part_at_index(&self, index: usize) -> T {
        let mut low = T::zero();
        if self.lower_bit_length > 0 {
            let offset = index * self.lower_bit_length;
            let mut remaining_bits = self.lower_bit_length;
            let mut bit_offset = 0;

            while remaining_bits > 0 {
                let bits_to_load = std::cmp::min(remaining_bits, 64);
                // TODO(tyb): error if chunk_start >= self.lower_bits_slice.len()
                // error if chunk_start + bits_to_load > self.lower_bits_slice.len()
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

impl<'a, T: CompressionInt> Iterator for EliasFanoMMapDecodingIterator<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        let was_post_skip = self.post_skip;

        // Check if the iterator was just used for a skip/peek
        if self.post_skip {
            // Advance past the '1' bit of the element we skipped to
            if self.cur_upper_bit_index < self.upper_bits_slice.len()
                && self.upper_bits_slice[self.cur_upper_bit_index]
            {
                self.cur_upper_bit_index += 1;
            }
            // Move to next element since skip_to() positioned us at the current element
            self.cur_elem_index += 1;
        }
        // Always consume the flag on a call to next()
        self.post_skip = false;

        if self.cur_elem_index < self.num_elem {
            self.decode_upper_part();
            let upper = self.cumulative_gap_sum;
            let lower = self.get_current_lower_part();

            // Only increment cur_elem_index at the end if we weren't in post_skip state
            // (which already incremented it before decoding)
            if !was_post_skip {
                self.cur_elem_index += 1;
            }

            Some((upper << self.lower_bit_length) | lower)
        } else {
            None
        }
    }
}
