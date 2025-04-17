use std::{clone, collections::HashMap};
use log::{info, error};
use proto::muopdb;
use proto::muopdb::{DocumentAttribute, attribute_value::Value};

#[derive(PartialEq, Debug)]
pub enum AttributeType {
    None,
    Integer,
    Float,
    String,
    Boolean,
    Text,
    Keyword,
    
    VectorInt,
    VectorKeyword,
}

pub struct AttributeSchema {
    fields: HashMap<String, AttributeType>,
}

impl AttributeSchema {
    pub fn new(mapping: HashMap<String, AttributeType>) -> Self {
        AttributeSchema {
            fields: mapping,
        }
    }

    pub fn new_from_proto(schema: muopdb::AttributeSchema) -> Self {
        let mut fields = HashMap::<String, AttributeType>::new();
        schema.attributes.into_iter().for_each( |attribute| {
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

    pub fn verify(&self, doc: DocumentAttribute) -> Result<bool, String> {
        for (key, attr_value) in doc.value.iter() {
            if !self.fields.contains_key(key) {
                return Err(format!("fields {} does not present in schema", key));
            }

            match attr_value.value.clone() {
                Some(v) => {
                    let mut doc_field = AttributeType::None;
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
                    info!("field {} does not set", key);
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
