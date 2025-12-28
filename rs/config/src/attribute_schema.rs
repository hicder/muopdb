use std::collections::HashMap;
use std::str::FromStr;

use proto::muopdb;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Copy, Default)]
pub enum Language {
    Arabic,
    Danish,
    Dutch,
    #[default]
    English,
    Finnish,
    French,
    German,
    Greek,
    Hungarian,
    Italian,
    Norwegian,
    Portuguese,
    Romanian,
    Russian,
    Spanish,
    Swedish,
    Tamil,
    Turkish,
    Vietnamese,
}

impl FromStr for Language {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "arabic" => Ok(Language::Arabic),
            "danish" => Ok(Language::Danish),
            "dutch" => Ok(Language::Dutch),
            "english" => Ok(Language::English),
            "finnish" => Ok(Language::Finnish),
            "french" => Ok(Language::French),
            "german" => Ok(Language::German),
            "greek" => Ok(Language::Greek),
            "hungarian" => Ok(Language::Hungarian),
            "italy" | "italian" => Ok(Language::Italian),
            "norwegian" => Ok(Language::Norwegian),
            "portuguese" => Ok(Language::Portuguese),
            "romanian" => Ok(Language::Romanian),
            "russian" => Ok(Language::Russian),
            "spanish" => Ok(Language::Spanish),
            "swedish" => Ok(Language::Swedish),
            "tamil" => Ok(Language::Tamil),
            "turkish" => Ok(Language::Turkish),
            "vietnamese" => Ok(Language::Vietnamese),
            _ => Err(format!("Invalid language: {}", s)),
        }
    }
}

impl Language {
    pub fn parse_str(s: &str) -> Option<Self> {
        Self::from_str(s).ok()
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum AttributeType {
    None,
    Integer,
    Float,
    Boolean,
    Text(Language),
    Keyword,

    VectorInt,
    VectorKeyword,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct AttributeSchema {
    pub fields: HashMap<String, AttributeType>,
}

impl From<muopdb::AttributeSchema> for AttributeSchema {
    fn from(value: muopdb::AttributeSchema) -> Self {
        let mut fields = HashMap::<String, AttributeType>::new();
        value.attributes.into_iter().for_each(|attribute| {
            let attribute_type = match attribute.r#type() {
                muopdb::AttributeType::Int => AttributeType::Integer,
                muopdb::AttributeType::Float => AttributeType::Float,
                muopdb::AttributeType::Bool => AttributeType::Boolean,
                muopdb::AttributeType::Keyword => AttributeType::Keyword,
                muopdb::AttributeType::Text => {
                    let language = attribute
                        .language
                        .as_deref()
                        .and_then(Language::parse_str)
                        .unwrap_or(Language::English);
                    AttributeType::Text(language)
                }
                muopdb::AttributeType::VectorInt => AttributeType::VectorInt,
                muopdb::AttributeType::VectorKeyword => AttributeType::VectorKeyword,
            };
            fields.insert(attribute.name.clone(), attribute_type);
        });
        Self::new(fields)
    }
}

impl AttributeSchema {
    pub fn new(mapping: HashMap<String, AttributeType>) -> Self {
        AttributeSchema { fields: mapping }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_trait_conversion() {
        let proto_attributes = vec![
            muopdb::AttributeField {
                name: String::from("i32_field"),
                r#type: muopdb::AttributeType::Int as i32,
                language: None,
            },
            muopdb::AttributeField {
                name: String::from("text_field"),
                r#type: muopdb::AttributeType::Text as i32,
                language: Some("french".to_string()),
            },
            muopdb::AttributeField {
                name: String::from("float_field"),
                r#type: muopdb::AttributeType::Float as i32,
                language: None,
            },
            muopdb::AttributeField {
                name: String::from("bool_field"),
                r#type: muopdb::AttributeType::Bool as i32,
                language: None,
            },
            muopdb::AttributeField {
                name: String::from("vector_int_field"),
                r#type: muopdb::AttributeType::VectorInt as i32,
                language: None,
            },
            muopdb::AttributeField {
                name: String::from("vector_keyword_field"),
                r#type: muopdb::AttributeType::VectorKeyword as i32,
                language: None,
            },
            muopdb::AttributeField {
                name: String::from("keyword_field"),
                r#type: muopdb::AttributeType::Keyword as i32,
                language: None,
            },
        ];

        let expected_schema_len = proto_attributes.len();
        let proto_schema = muopdb::AttributeSchema {
            attributes: proto_attributes,
        };

        let schema: AttributeSchema = proto_schema.into();

        assert_eq!(schema.fields.len(), expected_schema_len);
        assert_eq!(
            schema.fields.get("i32_field"),
            Some(&AttributeType::Integer)
        );
        assert_eq!(
            schema.fields.get("text_field"),
            Some(&AttributeType::Text(Language::French))
        );
        assert_eq!(
            schema.fields.get("keyword_field"),
            Some(&AttributeType::Keyword)
        );
        assert_eq!(
            schema.fields.get("float_field"),
            Some(&AttributeType::Float)
        );
        assert_eq!(
            schema.fields.get("bool_field"),
            Some(&AttributeType::Boolean)
        );
        assert_eq!(
            schema.fields.get("vector_int_field"),
            Some(&AttributeType::VectorInt)
        );
        assert_eq!(
            schema.fields.get("vector_keyword_field"),
            Some(&AttributeType::VectorKeyword)
        );
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
    fn test_language_parsing() {
        assert_eq!(Language::from_str("English"), Ok(Language::English));
        assert_eq!(Language::from_str("french"), Ok(Language::French));
        assert_eq!(Language::from_str("VIETNAMESE"), Ok(Language::Vietnamese));
        assert_eq!(
            Language::from_str("invalid"),
            Err("Invalid language: invalid".to_string())
        );
    }
}
