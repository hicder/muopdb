use serde::{Deserialize, Serialize};

// TODO(hicder): support more quantizers
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub enum QuantizerType {
    ProductQuantizer,
    #[default]
    NoQuantizer,
}

impl From<i32> for QuantizerType {
    fn from(value: i32) -> Self {
        match value {
            0 => QuantizerType::NoQuantizer,
            1 => QuantizerType::ProductQuantizer,
            _ => QuantizerType::NoQuantizer, // Default to NoQuantizer for unknown values
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub enum DistanceType {
    DotProduct,
    #[default]
    L2,
}

// TODO(tyb): support more encoding
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub enum IntSeqEncodingType {
    EliasFano,
    #[default]
    PlainEncoding,
}

impl From<i32> for IntSeqEncodingType {
    fn from(value: i32) -> Self {
        match value {
            0 => IntSeqEncodingType::PlainEncoding,
            1 => IntSeqEncodingType::EliasFano,
            _ => IntSeqEncodingType::PlainEncoding, // Default to PlainEncoding for unknown values
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub enum IndexType {
    Hnsw,
    Ivf,
    #[default]
    Spann,
}
