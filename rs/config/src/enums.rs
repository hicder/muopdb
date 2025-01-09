use serde::{Deserialize, Serialize};

// TODO(hicder): support more quantizers
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub enum QuantizerType {
    ProductQuantizer,
    #[default]
    NoQuantizer,
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

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub enum IndexType {
    Hnsw,
    Ivf,
    #[default]
    Spann,
}

