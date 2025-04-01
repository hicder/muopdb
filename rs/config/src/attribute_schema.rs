use std::collections::HashMap;
use proto::muopdb;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
