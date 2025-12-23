use proto::muopdb::DocumentAttribute;
use rkyv::util::AlignedVec;
use utils::mem::{transmute_u8_to_slice, transmute_u8_to_val_aligned};

#[derive(Debug, Clone)]
pub struct WalEntry {
    pub buffer: AlignedVec<16>,
    pub seq_no: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WalOpType<T> {
    Insert(T),
    Delete,
}

pub struct WalEntryDecoded<'a> {
    pub doc_ids: &'a [u128],
    pub user_ids: &'a [u128],
    pub op_type: WalOpType<&'a [f32]>,

    // None if this is delete operation
    pub attributes: Option<Vec<DocumentAttribute>>,
}

const ATTR_TYPE_INT: u8 = 0;
const ATTR_TYPE_FLOAT: u8 = 1;
const ATTR_TYPE_BOOL: u8 = 2;
const ATTR_TYPE_KEYWORD: u8 = 3;
const ATTR_TYPE_TEXT: u8 = 4;
const ATTR_TYPE_VECTOR_INT: u8 = 5;
const ATTR_TYPE_VECTOR_KEYWORD: u8 = 6;

pub fn serialize_document_attribute(attr: &DocumentAttribute) -> Vec<u8> {
    let mut result = Vec::new();

    for (key, value) in &attr.value {
        let key_bytes = key.as_bytes();

        result.extend_from_slice(&(key_bytes.len() as u32).to_le_bytes());
        result.extend_from_slice(key_bytes);

        match &value.value {
            Some(proto::muopdb::attribute_value::Value::IntValue(v)) => {
                result.push(ATTR_TYPE_INT);
                result.extend_from_slice(&v.to_le_bytes());
            }
            Some(proto::muopdb::attribute_value::Value::FloatValue(v)) => {
                result.push(ATTR_TYPE_FLOAT);
                result.extend_from_slice(&v.to_le_bytes());
            }
            Some(proto::muopdb::attribute_value::Value::BoolValue(v)) => {
                result.push(ATTR_TYPE_BOOL);
                result.push(if *v { 1 } else { 0 });
            }
            Some(proto::muopdb::attribute_value::Value::KeywordValue(v)) => {
                result.push(ATTR_TYPE_KEYWORD);
                let v_bytes = v.as_bytes();
                result.extend_from_slice(&(v_bytes.len() as u32).to_le_bytes());
                result.extend_from_slice(v_bytes);
            }
            Some(proto::muopdb::attribute_value::Value::TextValue(v)) => {
                result.push(ATTR_TYPE_TEXT);
                let v_bytes = v.as_bytes();
                result.extend_from_slice(&(v_bytes.len() as u32).to_le_bytes());
                result.extend_from_slice(v_bytes);
            }
            Some(proto::muopdb::attribute_value::Value::VectorIntValue(v)) => {
                result.push(ATTR_TYPE_VECTOR_INT);
                result.extend_from_slice(&(v.values.len() as u32).to_le_bytes());
                for val in &v.values {
                    result.extend_from_slice(&val.to_le_bytes());
                }
            }
            Some(proto::muopdb::attribute_value::Value::VectorKeywordValue(v)) => {
                result.push(ATTR_TYPE_VECTOR_KEYWORD);
                result.extend_from_slice(&(v.values.len() as u32).to_le_bytes());
                for val in &v.values {
                    let val_bytes = val.as_bytes();
                    result.extend_from_slice(&(val_bytes.len() as u32).to_le_bytes());
                    result.extend_from_slice(val_bytes);
                }
            }
            None => {
                result.push(ATTR_TYPE_INT);
                result.extend_from_slice(&0i64.to_le_bytes());
            }
        }
    }

    result
}

pub fn serialize_document_attributes(attrs: &[DocumentAttribute]) -> Vec<u8> {
    let mut result = Vec::new();
    for attr in attrs {
        result.push(1);
        result.extend_from_slice(&serialize_document_attribute(attr));
    }
    result
}

pub fn deserialize_document_attribute(data: &[u8]) -> Option<proto::muopdb::DocumentAttribute> {
    if data.is_empty() {
        return None;
    }

    let mut offset = 0;
    let mut attributes = std::collections::HashMap::new();

    while offset < data.len() {
        if offset + 4 > data.len() {
            break;
        }
        let key_len = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;

        if offset + key_len > data.len() {
            break;
        }
        let key = String::from_utf8_lossy(&data[offset..offset + key_len]).to_string();
        offset += key_len;

        if offset >= data.len() {
            break;
        }
        let attr_type = data[offset];
        offset += 1;

        let value = match attr_type {
            ATTR_TYPE_INT => {
                if offset + 8 > data.len() {
                    break;
                }
                let val = i64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
                offset += 8;
                proto::muopdb::AttributeValue {
                    value: Some(proto::muopdb::attribute_value::Value::IntValue(val)),
                }
            }
            ATTR_TYPE_FLOAT => {
                if offset + 4 > data.len() {
                    break;
                }
                let val = f32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
                offset += 4;
                proto::muopdb::AttributeValue {
                    value: Some(proto::muopdb::attribute_value::Value::FloatValue(val)),
                }
            }
            ATTR_TYPE_BOOL => {
                if offset >= data.len() {
                    break;
                }
                let val = data[offset] != 0;
                offset += 1;
                proto::muopdb::AttributeValue {
                    value: Some(proto::muopdb::attribute_value::Value::BoolValue(val)),
                }
            }
            ATTR_TYPE_KEYWORD => {
                if offset + 4 > data.len() {
                    break;
                }
                let val_len =
                    u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
                offset += 4;
                if offset + val_len > data.len() {
                    break;
                }
                let val = String::from_utf8_lossy(&data[offset..offset + val_len]).to_string();
                offset += val_len;
                proto::muopdb::AttributeValue {
                    value: Some(proto::muopdb::attribute_value::Value::KeywordValue(val)),
                }
            }
            ATTR_TYPE_TEXT => {
                if offset + 4 > data.len() {
                    break;
                }
                let val_len =
                    u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
                offset += 4;
                if offset + val_len > data.len() {
                    break;
                }
                let val = String::from_utf8_lossy(&data[offset..offset + val_len]).to_string();
                offset += val_len;
                proto::muopdb::AttributeValue {
                    value: Some(proto::muopdb::attribute_value::Value::TextValue(val)),
                }
            }
            ATTR_TYPE_VECTOR_INT => {
                if offset + 4 > data.len() {
                    break;
                }
                let count =
                    u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
                offset += 4;
                let mut values = Vec::with_capacity(count);
                for _ in 0..count {
                    if offset + 8 > data.len() {
                        break;
                    }
                    let val = i64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
                    values.push(val);
                    offset += 8;
                }
                proto::muopdb::AttributeValue {
                    value: Some(proto::muopdb::attribute_value::Value::VectorIntValue(
                        proto::muopdb::AttributeVectorIntValue { values },
                    )),
                }
            }
            ATTR_TYPE_VECTOR_KEYWORD => {
                if offset + 4 > data.len() {
                    break;
                }
                let count =
                    u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
                offset += 4;
                let mut values = Vec::with_capacity(count);
                for _ in 0..count {
                    if offset + 4 > data.len() {
                        break;
                    }
                    let val_len =
                        u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
                    offset += 4;
                    if offset + val_len > data.len() {
                        break;
                    }
                    let val = String::from_utf8_lossy(&data[offset..offset + val_len]).to_string();
                    values.push(val);
                    offset += val_len;
                }
                proto::muopdb::AttributeValue {
                    value: Some(proto::muopdb::attribute_value::Value::VectorKeywordValue(
                        proto::muopdb::AttributeVectorKeywordValue { values },
                    )),
                }
            }
            _ => {
                offset += 1;
                proto::muopdb::AttributeValue {
                    value: Some(proto::muopdb::attribute_value::Value::IntValue(0)),
                }
            }
        };

        attributes.insert(key, value);
    }

    Some(proto::muopdb::DocumentAttribute { value: attributes })
}

pub fn deserialize_document_attributes(data: &[u8], num_docs: usize) -> Vec<DocumentAttribute> {
    eprintln!(
        "DEBUG DESERIALIZE: data.len()={}, num_docs={}",
        data.len(),
        num_docs
    );
    let mut result = Vec::with_capacity(num_docs);
    let mut offset = 0;

    for _ in 0..num_docs {
        if offset >= data.len() {
            eprintln!("DEBUG DESERIALIZE: offset >= len, pushing None");
            result.push(DocumentAttribute::default());
            continue;
        }
        let has_attr = data[offset];
        eprintln!("DEBUG DESERIALIZE: has_attr={}", has_attr);
        offset += 1;
        if has_attr == 0 {
            result.push(DocumentAttribute::default());
        } else {
            let attr = deserialize_document_attribute(&data[offset..]);
            if let Some(ref a) = attr {
                offset += serialize_document_attribute(a).len();
            }
            result.push(attr.unwrap_or_default());
        }
    }

    result
}

impl WalEntry {
    pub fn decode(&self, num_features: usize) -> WalEntryDecoded<'_> {
        let length = self.buffer.len() - 1;
        let mut offset = 0;
        let num_docs =
            transmute_u8_to_val_aligned::<u64>(&self.buffer[offset..offset + 8]) as usize;
        offset += 8;
        let num_users =
            transmute_u8_to_val_aligned::<u64>(&self.buffer[offset..offset + 8]) as usize;
        offset += 8;
        let doc_ids = transmute_u8_to_slice::<u128>(&self.buffer[offset..offset + num_docs * 16]);
        offset += num_docs * 16;
        let user_ids = transmute_u8_to_slice::<u128>(&self.buffer[offset..offset + num_users * 16]);
        offset += num_users * 16;

        let is_delete = self.buffer[length] == 1;

        let (_, op_type) = if is_delete {
            (&[][..], WalOpType::Delete)
        } else {
            let data_len = num_features * num_docs * 4;
            let data = transmute_u8_to_slice::<f32>(&self.buffer[offset..offset + data_len]);
            offset += data_len;
            assert_eq!(
                data.len(),
                num_features * num_docs,
                "num_vectors mismatch while decoding WalEntry data"
            );
            (data, WalOpType::Insert(data))
        };

        // let _attr_len = if offset + 4 <= length {
        //     u32::from_le_bytes(self.buffer[offset..offset + 4].try_into().unwrap()) as usize
        // } else {
        //     0
        // };

        let attributes = if !is_delete {
            let attr_data_start = offset + 4;
            let attr_data_end = length - 1;
            if attr_data_end > attr_data_start {
                Some(deserialize_document_attributes(
                    &self.buffer[attr_data_start..attr_data_end],
                    num_docs,
                ))
            } else {
                let attrs = vec![DocumentAttribute::default(); num_docs];
                Some(attrs)
            }
        } else {
            None
        };

        WalEntryDecoded {
            doc_ids,
            user_ids,
            op_type,
            attributes,
        }
    }
}
