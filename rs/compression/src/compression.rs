use std::fs::File;
use std::io::BufWriter;

use anyhow::Result;
use log::warn;

pub trait CompressionInt:
    Copy                                    // Allows copying values instead of moving
    + std::fmt::Debug                       // For debugging and printing
    + std::fmt::Display                     // For debugging and printing
    + PartialOrd                            // For checking if sequence is sorted
    + From<u8>                              // Convert from u8 literal (e.g., 0, 1)
    + TryInto<u64>                          // Convert to u64 (may fail for large values)
    + TryFrom<u64>                          // Convert from u64 (may fail for large values)
    + std::ops::Shr<usize, Output = Self>   // (>>)
    + std::ops::Shl<usize, Output = Self>   // (<<)
    + std::ops::BitAnd<Self, Output = Self> // (&)
    + std::ops::BitOr<Self, Output = Self>  // (|)
    + std::ops::Sub<Self, Output = Self>    // (-)
    + std::ops::Add<Self, Output = Self>    // (+)
    + std::ops::Div<Self, Output = Self>    // (/)
    + std::ops::AddAssign<Self>             // (+=)
    + num_traits::ops::bytes::ToBytes       // to_le_bytes
    + Default                               // Default value (usually 0)
{
    // Required methods
    fn zero() -> Self;
    fn one() -> Self;
    fn max_value() -> Self;
    fn bits() -> usize;
    // Since Elias Fano uses BitVec<u64> under the hood, use u64 as intermediate type
    fn as_u64(&self) -> u64;
    fn from_u64(n: u64) -> Self;
    fn from_le_bytes(bytes: &[u8]) -> Result<Self>;
    fn leading_zeros(&self) -> u32;
}

impl CompressionInt for u32 {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
    fn max_value() -> Self {
        u32::MAX
    }
    fn bits() -> usize {
        32
    }
    fn as_u64(&self) -> u64 {
        *self as u64
    }
    fn from_u64(n: u64) -> Self {
        if n > u32::MAX as u64 {
            warn!("u64 value {n} too large for u32, saturating at u32::MAX");
            u32::MAX
        } else {
            n as u32
        }
    }
    fn from_le_bytes(bytes: &[u8]) -> Result<Self> {
        let array: [u8; 4] = bytes.try_into()?;
        Ok(u32::from_le_bytes(array))
    }
    fn leading_zeros(&self) -> u32 {
        u32::leading_zeros(*self)
    }
}

impl CompressionInt for u64 {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
    fn max_value() -> Self {
        u64::MAX
    }
    fn bits() -> usize {
        64
    }
    fn as_u64(&self) -> u64 {
        *self
    }
    fn from_u64(n: u64) -> Self {
        n
    }
    fn from_le_bytes(bytes: &[u8]) -> Result<Self> {
        let array: [u8; 8] = bytes.try_into()?;
        Ok(u64::from_le_bytes(array))
    }
    fn leading_zeros(&self) -> u32 {
        u64::leading_zeros(*self)
    }
}

impl CompressionInt for u128 {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
    fn max_value() -> Self {
        u128::MAX
    }
    fn bits() -> usize {
        128
    }
    fn as_u64(&self) -> u64 {
        if *self > u64::MAX as u128 {
            warn!("u128 value {self} too large for u64, truncating");
        }
        *self as u64
    }
    fn from_u64(n: u64) -> Self {
        n as u128
    }
    fn from_le_bytes(bytes: &[u8]) -> Result<Self> {
        let array: [u8; 16] = bytes.try_into()?;
        Ok(u128::from_le_bytes(array))
    }
    fn leading_zeros(&self) -> u32 {
        u128::leading_zeros(*self)
    }
}

pub trait IntSeqEncoder<T: CompressionInt = u64> {
    /// Creates an encoder
    fn new_encoder(universe: T, num_elem: usize) -> Self
    where
        Self: Sized;

    /// Compresses a sorted slice of integers
    fn encode_batch(&mut self, slice: &[T]) -> Result<()>;

    /// Compresses an u64 integer
    fn encode_value(&mut self, value: &T) -> Result<()>;

    /// Returns the size of the encoded data (that would be written to disk)
    fn len(&self) -> usize;

    /// Writes to disk and returns number of bytes written (which can be just len(),
    /// or more if extra info is also required for decoding)
    fn write(&self, writer: &mut BufWriter<&mut File>) -> Result<usize>;
}

pub trait IntSeqDecoder<T: CompressionInt = u64> {
    type IteratorType<'a>: Iterator<Item = T>
    where
        T: 'a;

    /// Creates a decoder
    fn new_decoder(byte_slice: &[u8]) -> Result<Self>
    where
        Self: Sized;

    /// Creates an iterator that iterates the encoded data and decodes one element at a time on the
    /// fly
    fn get_iterator<'a>(&self, byte_slice: &'a [u8]) -> Self::IteratorType<'a>
    where
        T: 'a;
}
