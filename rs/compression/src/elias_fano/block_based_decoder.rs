use std::marker::PhantomData;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use bitvec::prelude::*;
use utils::block_cache::cache::{BlockCache, FileId};
use utils::mem::transmute_u8_to_slice;

use crate::compression::{AsyncIntSeqDecoder, AsyncIntSeqIterator, CompressionInt};

pub struct BlockBasedEliasFanoDecoder<T: CompressionInt = u64> {
    block_cache: Arc<BlockCache>,
    file_id: FileId,
    base_offset: u64,
    buffer_size: usize,

    num_elem: usize,
    lower_bit_length: usize,
    upper_vec_len: usize,

    lower_bits_byte_offset: u64,
    upper_bits_byte_offset: u64,

    _phantom: PhantomData<T>,
}

#[async_trait::async_trait]
impl<T: CompressionInt + Send + Sync> AsyncIntSeqDecoder<T> for BlockBasedEliasFanoDecoder<T> {
    type IteratorType = BlockBasedEliasFanoIterator<T>;

    async fn new_decoder(
        block_cache: Arc<BlockCache>,
        file_id: FileId,
        offset: u64,
        buffer_size: usize,
    ) -> Result<Self> {
        let metadata_bytes = block_cache.read(file_id, offset, 32).await?;
        if metadata_bytes.len() < 32 {
            return Err(anyhow!("Not enough metadata for EliasFano encoded data"));
        }
        let metadata = transmute_u8_to_slice::<u64>(&metadata_bytes);

        let num_elem = metadata[0] as usize;
        let lower_bit_length = metadata[1] as usize;
        let lower_vec_len = metadata[2] as usize;
        let upper_vec_len = metadata[3] as usize;

        Ok(Self {
            block_cache,
            file_id,
            base_offset: offset,
            buffer_size,
            num_elem,
            lower_bit_length,
            upper_vec_len,
            lower_bits_byte_offset: 32,
            upper_bits_byte_offset: 32 + (lower_vec_len * 8) as u64,
            _phantom: PhantomData,
        })
    }

    fn into_iterator(self) -> Self::IteratorType {
        BlockBasedEliasFanoIterator {
            block_cache: self.block_cache,
            file_id: self.file_id,
            base_offset: self.base_offset,
            buffer_size: self.buffer_size,
            num_elem: self.num_elem,
            lower_bit_length: self.lower_bit_length,
            lower_bits_byte_offset: self.lower_bits_byte_offset,
            upper_bits_byte_offset: self.upper_bits_byte_offset,
            upper_vec_len: self.upper_vec_len,

            cur_elem_index: 0,
            cur_upper_bit_index: 0,
            cumulative_gap_sum: T::zero(),
            post_skip: false,

            upper_bits_buffer: Vec::new(),
            upper_bits_buffer_bit_start: 0,
        }
    }
}

pub struct BlockBasedEliasFanoIterator<T: CompressionInt = u64> {
    block_cache: Arc<BlockCache>,
    file_id: FileId,
    base_offset: u64,
    buffer_size: usize,
    num_elem: usize,
    lower_bit_length: usize,
    lower_bits_byte_offset: u64,
    upper_bits_byte_offset: u64,
    upper_vec_len: usize,

    cur_elem_index: usize,
    cur_upper_bit_index: usize,
    cumulative_gap_sum: T,
    post_skip: bool,

    upper_bits_buffer: Vec<u8>,
    upper_bits_buffer_bit_start: usize,
}

impl<T: CompressionInt> BlockBasedEliasFanoIterator<T> {
    async fn get_lower_part_at_index(&self, index: usize) -> Result<T> {
        if self.lower_bit_length == 0 {
            return Ok(T::zero());
        }

        let bit_offset = index * self.lower_bit_length;
        let byte_start = self.base_offset + self.lower_bits_byte_offset + (bit_offset / 8) as u64;
        let bit_in_first_byte = bit_offset % 8;

        let bytes_needed = (bit_in_first_byte + self.lower_bit_length).div_ceil(8);
        let data = self
            .block_cache
            .read(self.file_id, byte_start, bytes_needed as u64)
            .await?;

        let bits = BitSlice::<u8, Lsb0>::from_slice(&data);
        let mut low = T::zero();
        let mut remaining_bits = self.lower_bit_length;
        let mut bit_offset_in_low = 0;

        while remaining_bits > 0 {
            let bits_to_load = std::cmp::min(remaining_bits, 64);
            let chunk_start = bit_in_first_byte + bit_offset_in_low;
            let chunk = bits[chunk_start..chunk_start + bits_to_load].load::<u64>();
            low = low | (T::from_u64(chunk) << bit_offset_in_low);
            bit_offset_in_low += bits_to_load;
            remaining_bits -= bits_to_load;
        }

        Ok(low)
    }

    async fn ensure_upper_bits_buffered(&mut self, bit_index: usize) -> Result<()> {
        let buffer_bit_len = self.upper_bits_buffer.len() * 8;
        let buffer_bit_end = self.upper_bits_buffer_bit_start + buffer_bit_len;

        if bit_index >= self.upper_bits_buffer_bit_start && bit_index < buffer_bit_end {
            return Ok(());
        }

        let byte_offset = self.base_offset + self.upper_bits_byte_offset + (bit_index / 8) as u64;
        let total_upper_bytes = (self.upper_vec_len * 8) as u64;
        let bytes_from_bit = (bit_index / 8) as u64;

        if bytes_from_bit >= total_upper_bytes {
            return Err(anyhow!("Upper bit index out of bounds"));
        }

        let remaining_bytes = total_upper_bytes - bytes_from_bit;
        let read_len = std::cmp::min(self.buffer_size as u64, remaining_bytes);

        self.upper_bits_buffer = self
            .block_cache
            .read(self.file_id, byte_offset, read_len)
            .await?;
        self.upper_bits_buffer_bit_start = (bit_index / 8) * 8;

        Ok(())
    }

    fn get_upper_bit(&self, bit_index: usize) -> bool {
        let local_bit = bit_index - self.upper_bits_buffer_bit_start;
        let byte_idx = local_bit / 8;
        let bit_in_byte = local_bit % 8;
        (self.upper_bits_buffer[byte_idx] >> bit_in_byte) & 1 == 1
    }

    async fn decode_upper_part(&mut self) -> Result<()> {
        let max_bit = self.upper_vec_len * 64;

        while self.cur_upper_bit_index < max_bit {
            self.ensure_upper_bits_buffered(self.cur_upper_bit_index)
                .await?;

            if self.get_upper_bit(self.cur_upper_bit_index) {
                self.cur_upper_bit_index += 1;
                break;
            }

            self.cumulative_gap_sum += T::one();
            self.cur_upper_bit_index += 1;
        }

        Ok(())
    }

    async fn get_value_at_index(&mut self, index: usize) -> Result<Option<T>> {
        if index >= self.num_elem {
            return Ok(None);
        }

        let mut high = T::zero();
        let mut pos = 0;
        let mut elements_seen = 0;
        let max_bit = self.upper_vec_len * 64;

        while pos < max_bit && elements_seen <= index {
            self.ensure_upper_bits_buffered(pos).await?;
            if self.get_upper_bit(pos) {
                if elements_seen == index {
                    break;
                }
                elements_seen += 1;
            } else {
                high += T::one();
            }
            pos += 1;
        }

        let low = self.get_lower_part_at_index(index).await?;
        Ok(Some((high << self.lower_bit_length) | low))
    }

    async fn seek_to_index(&mut self, target_index: usize) -> Result<()> {
        if target_index >= self.num_elem {
            self.cur_elem_index = self.num_elem;
            return Ok(());
        }

        self.cur_upper_bit_index = 0;
        self.cumulative_gap_sum = T::zero();
        self.cur_elem_index = 0;

        while self.cur_elem_index < target_index {
            self.decode_upper_part().await?;
            self.cur_elem_index += 1;
        }

        let max_bit = self.upper_vec_len * 64;
        while self.cur_upper_bit_index < max_bit {
            self.ensure_upper_bits_buffered(self.cur_upper_bit_index)
                .await?;
            if self.get_upper_bit(self.cur_upper_bit_index) {
                break;
            }
            self.cumulative_gap_sum += T::one();
            self.cur_upper_bit_index += 1;
        }

        self.post_skip = true;
        Ok(())
    }
}

#[async_trait::async_trait]
impl<T: CompressionInt + Send + Sync> AsyncIntSeqIterator<T> for BlockBasedEliasFanoIterator<T> {
    async fn next(&mut self) -> Result<Option<T>> {
        let was_post_skip = self.post_skip;

        if self.post_skip {
            let max_bit = self.upper_vec_len * 64;
            if self.cur_upper_bit_index < max_bit {
                self.ensure_upper_bits_buffered(self.cur_upper_bit_index)
                    .await?;
                if self.get_upper_bit(self.cur_upper_bit_index) {
                    self.cur_upper_bit_index += 1;
                }
            }
            self.cur_elem_index += 1;
        }
        self.post_skip = false;

        if self.cur_elem_index < self.num_elem {
            self.decode_upper_part().await?;
            let upper = self.cumulative_gap_sum;
            let lower = self.get_lower_part_at_index(self.cur_elem_index).await?;

            if !was_post_skip {
                self.cur_elem_index += 1;
            }

            Ok(Some((upper << self.lower_bit_length) | lower))
        } else {
            Ok(None)
        }
    }

    async fn current(&mut self) -> Result<Option<T>> {
        self.get_value_at_index(self.cur_elem_index).await
    }

    async fn skip_to(&mut self, target: T) -> Result<()> {
        if self.cur_elem_index >= self.num_elem {
            return Ok(());
        }

        let max_possible_high = T::from_u64((self.upper_vec_len * 64) as u64);
        let max_possible_value = (max_possible_high << self.lower_bit_length)
            | ((T::one() << self.lower_bit_length) - T::one());
        if target > max_possible_value {
            self.cur_elem_index = self.num_elem;
            return Ok(());
        }

        let mut left = self.cur_elem_index;
        let mut right = self.num_elem;
        let mut result_index = None;

        while left < right {
            let mid = left + (right - left) / 2;

            if let Some(mid_value) = self.get_value_at_index(mid).await? {
                if mid_value >= target {
                    result_index = Some(mid);
                    right = mid;
                } else {
                    left = mid + 1;
                }
            } else {
                break;
            }
        }

        if let Some(index) = result_index {
            self.seek_to_index(index).await?;
        } else {
            self.cur_elem_index = self.num_elem;
        }
        Ok(())
    }

    fn position(&self) -> usize {
        self.cur_elem_index
    }

    fn is_exhausted(&self) -> bool {
        self.cur_elem_index >= self.num_elem
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::BufWriter;

    use tempdir::TempDir;
    use utils::block_cache::cache::{BlockCache, BlockCacheConfig};

    use super::*;
    use crate::compression::IntSeqEncoder;
    use crate::elias_fano::ef::EliasFano;

    #[tokio::test]
    async fn test_block_based_elias_fano_decoding() {
        let values = vec![5u64, 8, 8, 15, 32];
        let upper_bound = 36u64;
        let mut ef = EliasFano::new_encoder(upper_bound, values.len());
        assert!(ef.encode_batch(&values).is_ok());

        let temp_dir = TempDir::new("test_block_based_elias_fano_decoding").unwrap();
        let file_path = temp_dir.path().join("test_file");
        let mut file = File::create(&file_path).unwrap();
        let mut writer = BufWriter::new(&mut file);
        assert!(ef.write(&mut writer).is_ok());
        drop(writer);

        let config = BlockCacheConfig::default();
        let block_cache = Arc::new(BlockCache::new(config));
        let file_id = block_cache
            .open_file(file_path.to_str().unwrap())
            .await
            .unwrap();

        let decoder =
            BlockBasedEliasFanoDecoder::<u64>::new_decoder(block_cache.clone(), file_id, 0, 4096)
                .await
                .unwrap();

        let mut iter = decoder.into_iterator();
        for expected in values {
            let actual = iter.next().await.unwrap().unwrap();
            assert_eq!(expected, actual);
        }
        assert!(iter.next().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_block_based_skip_to() {
        let values = vec![1u64, 5, 10, 15, 20, 25, 30];
        let upper_bound = 100u64;
        let mut ef = EliasFano::new_encoder(upper_bound, values.len());
        assert!(ef.encode_batch(&values).is_ok());

        let temp_dir = TempDir::new("test_block_based_skip_to").unwrap();
        let file_path = temp_dir.path().join("test_file");
        let mut file = File::create(&file_path).unwrap();
        let mut writer = BufWriter::new(&mut file);
        assert!(ef.write(&mut writer).is_ok());
        drop(writer);

        let config = BlockCacheConfig::default();
        let block_cache = Arc::new(BlockCache::new(config));
        let file_id = block_cache
            .open_file(file_path.to_str().unwrap())
            .await
            .unwrap();

        let decoder =
            BlockBasedEliasFanoDecoder::<u64>::new_decoder(block_cache.clone(), file_id, 0, 4096)
                .await
                .unwrap();

        let mut iter = decoder.into_iterator();

        // Skip to 12, should land on 15
        iter.skip_to(12).await.unwrap();
        assert_eq!(iter.current().await.unwrap(), Some(15));
        assert_eq!(iter.next().await.unwrap(), Some(20));

        // Skip to 25, should land on 25
        iter.skip_to(25).await.unwrap();
        assert_eq!(iter.current().await.unwrap(), Some(25));

        // Skip to 30, should land on 30
        iter.skip_to(30).await.unwrap();
        assert_eq!(iter.current().await.unwrap(), Some(30));

        // Skip beyond end
        iter.skip_to(100).await.unwrap();
        assert!(iter.is_exhausted());
    }

    #[tokio::test]
    async fn test_block_based_skip_to_more() {
        let values = vec![1u64, 5, 10, 15, 20, 25, 30];
        let upper_bound = 100u64;
        let mut ef = EliasFano::new_encoder(upper_bound, values.len());
        assert!(ef.encode_batch(&values).is_ok());

        let temp_dir = TempDir::new("test_block_based_skip_to_more").unwrap();
        let file_path = temp_dir.path().join("test_file");
        let mut file = File::create(&file_path).unwrap();
        let mut writer = BufWriter::new(&mut file);
        assert!(ef.write(&mut writer).is_ok());
        drop(writer);

        let config = BlockCacheConfig::default();
        let block_cache = Arc::new(BlockCache::new(config));
        let file_id = block_cache
            .open_file(file_path.to_str().unwrap())
            .await
            .unwrap();

        let decoder =
            BlockBasedEliasFanoDecoder::<u64>::new_decoder(block_cache.clone(), file_id, 0, 4096)
                .await
                .unwrap();

        let mut iter = decoder.into_iterator();

        // Skip to before first element
        iter.skip_to(0).await.unwrap();
        assert_eq!(iter.current().await.unwrap(), Some(1));
        assert_eq!(iter.position(), 0);

        // Skip to exactly first element
        iter.skip_to(1).await.unwrap();
        assert_eq!(iter.current().await.unwrap(), Some(1));

        // Skip to 12, should land on 15
        iter.skip_to(12).await.unwrap();
        assert_eq!(iter.current().await.unwrap(), Some(15));

        // Call next(), should advance to 20
        assert_eq!(iter.next().await.unwrap(), Some(20));

        // Skip to 25, should land on 25
        iter.skip_to(25).await.unwrap();
        assert_eq!(iter.current().await.unwrap(), Some(25));
    }

    #[tokio::test]
    async fn test_block_based_large_sequence() {
        // Test with larger sequence (> 64 elements to trigger binary search)
        let values: Vec<u64> = (1..=200).collect();
        let upper_bound = 500u64;

        let mut ef = EliasFano::new_encoder(upper_bound, values.len());
        assert!(ef.encode_batch(&values).is_ok());

        let temp_dir = TempDir::new("test_block_based_large_sequence").unwrap();
        let file_path = temp_dir.path().join("test_file");
        let mut file = File::create(&file_path).unwrap();
        let mut writer = BufWriter::new(&mut file);
        assert!(ef.write(&mut writer).is_ok());
        drop(writer);

        let config = BlockCacheConfig::default();
        let block_cache = Arc::new(BlockCache::new(config));
        let file_id = block_cache
            .open_file(file_path.to_str().unwrap())
            .await
            .unwrap();

        let decoder =
            BlockBasedEliasFanoDecoder::<u64>::new_decoder(block_cache.clone(), file_id, 0, 4096)
                .await
                .unwrap();

        let mut iter = decoder.into_iterator();

        // Skip to various positions
        iter.skip_to(50u64).await.unwrap();
        assert_eq!(iter.current().await.unwrap(), Some(50));

        iter.skip_to(150).await.unwrap();
        assert_eq!(iter.current().await.unwrap(), Some(150));

        iter.skip_to(200).await.unwrap();
        assert_eq!(iter.current().await.unwrap(), Some(200));
    }

    #[tokio::test]
    async fn test_block_based_single_element() {
        let values = vec![42u64];
        let upper_bound = 100u64;

        let mut ef = EliasFano::new_encoder(upper_bound, values.len());
        assert!(ef.encode_batch(&values).is_ok());

        let temp_dir = TempDir::new("test_block_based_single_element").unwrap();
        let file_path = temp_dir.path().join("test_file");
        let mut file = File::create(&file_path).unwrap();
        let mut writer = BufWriter::new(&mut file);
        assert!(ef.write(&mut writer).is_ok());
        drop(writer);

        let config = BlockCacheConfig::default();
        let block_cache = Arc::new(BlockCache::new(config));
        let file_id = block_cache
            .open_file(file_path.to_str().unwrap())
            .await
            .unwrap();

        let decoder =
            BlockBasedEliasFanoDecoder::<u64>::new_decoder(block_cache.clone(), file_id, 0, 4096)
                .await
                .unwrap();

        let mut iter = decoder.into_iterator();

        iter.skip_to(10u64).await.unwrap();
        assert_eq!(iter.current().await.unwrap(), Some(42));

        iter.skip_to(42).await.unwrap();
        assert_eq!(iter.current().await.unwrap(), Some(42));

        iter.skip_to(100).await.unwrap();
        assert_eq!(iter.current().await.unwrap(), None);
    }

    #[tokio::test]
    async fn test_block_based_u32_decoding() {
        let values = vec![10u32, 20, 30, 40, 50];
        let upper_bound = 100u32;
        let mut ef = EliasFano::<u32>::new_encoder(upper_bound, values.len());
        assert!(ef.encode_batch(&values).is_ok());

        let temp_dir = TempDir::new("test_block_based_u32_decoding").unwrap();
        let file_path = temp_dir.path().join("test_file");
        let mut file = File::create(&file_path).unwrap();
        let mut writer = BufWriter::new(&mut file);
        assert!(ef.write(&mut writer).is_ok());
        drop(writer);

        let config = BlockCacheConfig::default();
        let block_cache = Arc::new(BlockCache::new(config));
        let file_id = block_cache
            .open_file(file_path.to_str().unwrap())
            .await
            .unwrap();

        let decoder =
            BlockBasedEliasFanoDecoder::<u32>::new_decoder(block_cache.clone(), file_id, 0, 4096)
                .await
                .unwrap();

        let mut iter = decoder.into_iterator();
        for expected in values {
            let actual = iter.next().await.unwrap().unwrap();
            assert_eq!(expected, actual);
        }
    }

    #[tokio::test]
    async fn test_block_based_u128_decoding() {
        let values = vec![
            u64::MAX as u128,
            u64::MAX as u128 + 1,
            u64::MAX as u128 + 100,
        ];
        let upper_bound = u64::MAX as u128 + 200;
        let mut ef = EliasFano::<u128>::new_encoder(upper_bound, values.len());
        assert!(ef.encode_batch(&values).is_ok());

        let temp_dir = TempDir::new("test_block_based_u128_decoding").unwrap();
        let file_path = temp_dir.path().join("test_file");
        let mut file = File::create(&file_path).unwrap();
        let mut writer = BufWriter::new(&mut file);
        assert!(ef.write(&mut writer).is_ok());
        drop(writer);

        let config = BlockCacheConfig::default();
        let block_cache = Arc::new(BlockCache::new(config));
        let file_id = block_cache
            .open_file(file_path.to_str().unwrap())
            .await
            .unwrap();

        let decoder =
            BlockBasedEliasFanoDecoder::<u128>::new_decoder(block_cache.clone(), file_id, 0, 4096)
                .await
                .unwrap();

        let mut iter = decoder.into_iterator();
        for expected in values {
            let actual = iter.next().await.unwrap().unwrap();
            assert_eq!(expected, actual);
        }
    }
}
