use std::{clone, collections::HashMap};
use log::{debug};
use proto::muopdb;
use proto::muopdb::{DocumentAttribute, attribute_value::Value};
use serde::{Deserialize, Serialize};

#[derive(PartialEq, Debug, Serialize, Deserialize, Clone)]
pub enum AttributeType {
    None,
    Integer,
    Float,
    Boolean,
    Text,
    Keyword,
    
    VectorInt,
    VectorKeyword,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct AttributeSchema {
    fields: HashMap<String, AttributeType>,
}

impl From<muopdb::AttributeSchema> for AttributeSchema {
    fn from(value: muopdb::AttributeSchema) -> Self {
        let mut fields = HashMap::<String, AttributeType>::new();
        value.attributes.into_iter().for_each( |attribute| {
            let attribute_type;
            match attribute.r#type() {
                muopdb::AttributeType::Int => attribute_type = AttributeType::Integer,
                muopdb::AttributeType::Float => attribute_type = AttributeType::Float,
                muopdb::AttributeType::Bool => attribute_type = AttributeType::Boolean,
                muopdb::AttributeType::Keyword => attribute_type = AttributeType::Keyword,
                muopdb::AttributeType::Text => attribute_type = AttributeType::Text,
                muopdb::AttributeType::VectorInt => attribute_type = AttributeType::VectorInt,
                muopdb::AttributeType::VectorKeyword => attribute_type = AttributeType::VectorKeyword,
            }
            fields.insert(attribute.name.clone(), attribute_type);
        });
        Self::new(fields)
    }
}

impl AttributeSchema {
    pub fn new(mapping: HashMap<String, AttributeType>) -> Self {
        AttributeSchema {
            fields: mapping,
        }
    }

    pub fn verify(&self, doc: DocumentAttribute) -> Result<bool, String> {
        for (key, attr_value) in doc.value.iter() {
            if !self.fields.contains_key(key) {
                return Err(format!("fields {} does not present in schema", key));
            }

            match attr_value.value.clone() {
                Some(v) => {
                    let doc_field;
                    match v {
                        Value::IntValue(_) => doc_field = AttributeType::Integer,
                        Value::FloatValue(_) => doc_field = AttributeType::Float,
                        Value::BoolValue(_) => doc_field = AttributeType::Boolean,
                        Value::KeywordValue(_) => doc_field = AttributeType::Keyword,
                        Value::TextValue(_) => doc_field = AttributeType::Text,
                        Value::VectorIntValue(_) => doc_field = AttributeType::VectorInt,
                        Value::VectorKeywordValue(_) => doc_field = AttributeType::VectorKeyword, 
                    }
                    let schema_field_type = self.fields.get(key).unwrap();
                    if &doc_field != schema_field_type {
                       return Err(format!("fields {} does not match schema type, expected {:?}, found {:?}", key, schema_field_type, doc_field))
                    }
                },
                None => {
                    debug!("field {} does not set", key);
                }
            
            }
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proto::muopdb::attribute_value::Value;
    use proto::muopdb::DocumentAttribute;

    #[test]
    fn test_from_trait_conversion() {
        let mut proto_attributes = Vec::new();
        proto_attributes.push(muopdb::AttributeField {
            name: "i32_field".to_string(),
            r#type: muopdb::AttributeType::Int as i32,
        });
        proto_attributes.push(muopdb::AttributeField {
            name: String::from("text_field"),
            r#type: muopdb::AttributeType::Text as i32,
        });
        proto_attributes.push(muopdb::AttributeField { 
            name: String::from("float_field"),
            r#type: muopdb::AttributeType::Float as i32,
        });
        proto_attributes.push(muopdb::AttributeField { 
            name: String::from("bool_field"), 
            r#type: muopdb::AttributeType::Bool as i32, 
        });
        proto_attributes.push(muopdb::AttributeField { 
            name: String::from("vector_int_field"), 
            r#type: muopdb::AttributeType::VectorInt as i32, 
        });
        proto_attributes.push(muopdb::AttributeField { 
            name: String::from("vector_keyword_field"),
            r#type: muopdb::AttributeType::VectorKeyword as i32,
        });
        proto_attributes.push(muopdb::AttributeField { 
            name: String::from("keyword_field"), 
            r#type: muopdb::AttributeType::Keyword as i32, 
        });

        let expected_schema_len = proto_attributes.len();
        let proto_schema = muopdb::AttributeSchema {
            attributes: proto_attributes,
        };

        let schema: AttributeSchema = proto_schema.into();

        assert_eq!(schema.fields.len(), expected_schema_len);
        assert_eq!(schema.fields.get("i32_field"), Some(&AttributeType::Integer));
        assert_eq!(schema.fields.get("text_field"), Some(&AttributeType::Text));
        assert_eq!(schema.fields.get("keyword_field"), Some(&AttributeType::Keyword));
        assert_eq!(schema.fields.get("float_field"), Some(&AttributeType::Float));
        assert_eq!(schema.fields.get("bool_field"), Some(&AttributeType::Boolean));
        assert_eq!(schema.fields.get("vector_int_field"), Some(&AttributeType::VectorInt));
        assert_eq!(schema.fields.get("vector_keyword_field"), Some(&AttributeType::VectorKeyword));

    }

    #[test]
    fn test_from_trait_empty_schema() {
        let proto_schema = muopdb::AttributeSchema {
            attributes: Vec::new(),
        };

        let schema: AttributeSchema = proto_schema.into();

        assert!(schema.fields.is_empty());
    }
    #[test]
    fn test_verify_success() {
        let mut schema_mapping = HashMap::new();
        schema_mapping.insert("field1".to_string(), AttributeType::Integer);
        schema_mapping.insert("field2".to_string(), AttributeType::Text);

        let schema = AttributeSchema::new(schema_mapping);

        let mut doc_values = HashMap::new();
        doc_values.insert(
            "field1".to_string(),
            proto::muopdb::AttributeValue {
                value: Some(Value::IntValue(42)),
            },
        );
        doc_values.insert(
            "field2".to_string(),
            proto::muopdb::AttributeValue {
                value: Some(Value::TextValue("example".to_string())),
            },
        );

        let doc = DocumentAttribute { value: doc_values };

        assert!(schema.verify(doc).is_ok());
    }

    #[test]
    fn test_verify_missing_field() {
        let mut schema_mapping = HashMap::new();
        schema_mapping.insert("field1".to_string(), AttributeType::Integer);

        let schema = AttributeSchema::new(schema_mapping);

        let mut doc_values = HashMap::new();
        doc_values.insert(
            "field2".to_string(),
            proto::muopdb::AttributeValue {
                value: Some(Value::TextValue("example".to_string())),
            },
        );

        let doc = DocumentAttribute { value: doc_values };

        let result = schema.verify(doc);
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            "fields field2 does not present in schema"
        );
    }

    #[test]
    fn test_verify_type_mismatch() {
        let mut schema_mapping = HashMap::new();
        schema_mapping.insert("field1".to_string(), AttributeType::Integer);

        let schema = AttributeSchema::new(schema_mapping);

        let mut doc_values = HashMap::new();
        doc_values.insert(
            "field1".to_string(),
            proto::muopdb::AttributeValue {
                value: Some(Value::TextValue("example".to_string())),
            },
        );

        let doc = DocumentAttribute { value: doc_values };

        let result = schema.verify(doc);
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            "fields field1 does not match schema type, expected Integer, found Text"
        );
    }
}
