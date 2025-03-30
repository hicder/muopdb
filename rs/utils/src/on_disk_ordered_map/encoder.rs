use integer_encoding::VarInt;

pub struct DecodingResult<T> {
    pub value: T,
    pub num_bytes_read: usize,
}

pub trait IntegerCodec {
    fn id(&self) -> u8;

    fn encode_u64(&self, value: u64, buf: &mut [u8]) -> usize;
    fn decode_u64(&self, buf: &[u8]) -> DecodingResult<u64>;

    fn encode_u32(&self, value: u32, buf: &mut [u8]) -> usize;
    fn decode_u32(&self, buf: &[u8]) -> DecodingResult<u32>;
}

pub struct FixedIntegerCodec {}

impl IntegerCodec for FixedIntegerCodec {
    fn id(&self) -> u8 {
        0
    }

    fn encode_u64(&self, value: u64, buf: &mut [u8]) -> usize {
        buf[..8].copy_from_slice(&value.to_le_bytes());
        8
    }

    fn decode_u64(&self, buf: &[u8]) -> DecodingResult<u64> {
        DecodingResult {
            value: u64::from_le_bytes(buf[..8].try_into().unwrap()),
            num_bytes_read: 8,
        }
    }

    fn encode_u32(&self, value: u32, buf: &mut [u8]) -> usize {
        buf[..4].copy_from_slice(&value.to_le_bytes());
        4
    }

    fn decode_u32(&self, buf: &[u8]) -> DecodingResult<u32> {
        DecodingResult {
            value: u32::from_le_bytes(buf[..4].try_into().unwrap()),
            num_bytes_read: 4,
        }
    }
}

pub struct VarintIntegerCodec {}

impl IntegerCodec for VarintIntegerCodec {
    fn id(&self) -> u8 {
        1
    }

    #[inline(always)]
    fn encode_u64(&self, value: u64, buf: &mut [u8]) -> usize {
        value.encode_var(buf)
    }

    #[inline(always)]
    fn decode_u64(&self, buf: &[u8]) -> DecodingResult<u64> {
        let (value, num_bytes_read) = u64::decode_var(buf).unwrap();
        DecodingResult {
            value,
            num_bytes_read,
        }
    }

    #[inline(always)]
    fn encode_u32(&self, value: u32, buf: &mut [u8]) -> usize {
        value.encode_var(buf)
    }

    #[inline(always)]
    fn decode_u32(&self, buf: &[u8]) -> DecodingResult<u32> {
        let (value, num_bytes_read) = u32::decode_var(buf).unwrap();
        DecodingResult {
            value,
            num_bytes_read,
        }
    }
}
