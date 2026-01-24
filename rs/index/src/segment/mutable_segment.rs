use std::sync::atomic::AtomicU64;
use std::time::Instant;

use anyhow::{Ok, Result};
use config::attribute_schema::{AttributeSchema, AttributeType};
use config::collection::CollectionConfig;
use proto::muopdb::DocumentAttribute;
use tracing::info;

use crate::multi_spann::builder::MultiSpannBuilder;
use crate::multi_spann::writer::MultiSpannWriter;
use crate::multi_terms::builder::MultiTermBuilder;
use crate::multi_terms::writer::MultiTermWriter;
use crate::tokenizer::stemming_tokenizer::StemmingTokenizer;
use crate::tokenizer::{TokenStream, Tokenizer};

pub struct MutableSegment {
    multi_spann_builder: MultiSpannBuilder,
    multi_term_builder: MultiTermBuilder,
    attribute_schema: Option<AttributeSchema>,

    // Prevent a mutable segment from being modified after it is built.
    finalized: bool,
    last_sequence_number: AtomicU64,
    num_docs: AtomicU64,
    created_at: Instant,
}

impl MutableSegment {
    pub fn new(config: CollectionConfig, base_directory: String) -> Result<Self> {
        let term_directory = format!("{}/terms", base_directory);
        std::fs::create_dir_all(&term_directory)?;

        Ok(Self {
            multi_spann_builder: MultiSpannBuilder::new(config.clone(), base_directory)?,
            multi_term_builder: MultiTermBuilder::new(),
            attribute_schema: config.attribute_schema,
            finalized: false,
            last_sequence_number: AtomicU64::new(0),
            num_docs: AtomicU64::new(0),
            created_at: Instant::now(),
        })
    }

    pub fn created_at(&self) -> Instant {
        self.created_at
    }

    pub fn insert(&self, doc_id: u128, data: &[f32]) -> Result<()> {
        self.insert_for_user(0, doc_id, data, 0, DocumentAttribute::default())
    }

    /// Insert a document for a user
    pub fn insert_for_user(
        &self,
        user_id: u128,
        doc_id: u128,
        data: &[f32],
        sequence_number: u64,
        document_attribute: DocumentAttribute,
    ) -> Result<()> {
        info!(
            "Inserting for user: {user_id}, doc_id: {doc_id}, sequence_number: {sequence_number}"
        );
        if self.finalized {
            return Err(anyhow::anyhow!("Cannot insert into a finalized segment"));
        }

        let point_id = self.multi_spann_builder.insert(user_id, doc_id, data)?;

        // Process document attributes if present
        for (attr_name, attr_value) in document_attribute.value {
            match attr_value.value {
                Some(proto::muopdb::attribute_value::Value::TextValue(text)) => {
                    // Tokenize the text attribute
                    let language = self
                        .attribute_schema
                        .as_ref()
                        .and_then(|s| s.fields.get(&attr_name))
                        .map(|t| match t {
                            AttributeType::Text(l) => *l,
                            _ => config::attribute_schema::Language::English,
                        })
                        .unwrap_or(config::attribute_schema::Language::English);

                    let stemming_tokenizer = StemmingTokenizer::for_language(language);
                    let mut token_stream = stemming_tokenizer.input(&text);

                    // Process each token and insert with the term builder
                    while let Some(token) = token_stream.next() {
                        let term = format!("{}:{}", attr_name, token.text);
                        self.multi_term_builder.add(user_id, point_id, term)?;
                    }
                }
                Some(proto::muopdb::attribute_value::Value::KeywordValue(keyword)) => {
                    // For keyword attributes, insert the whole keyword as a single term
                    let term = format!("{}:{}", attr_name, keyword);
                    self.multi_term_builder.add(user_id, point_id, term)?;
                }
                _ => {
                    // Other attribute types (int, float, bool) are not tokenized
                    continue;
                }
            }
        }

        self.last_sequence_number
            .store(sequence_number, std::sync::atomic::Ordering::SeqCst);
        self.num_docs
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    pub fn is_valid_doc_id(&self, user_id: u128, doc_id: u128) -> bool {
        self.multi_spann_builder.is_valid_doc_id(user_id, doc_id)
    }

    pub fn invalidate(&self, user_id: u128, doc_id: u128, sequence_number: u64) -> Result<bool> {
        self.last_sequence_number
            .store(sequence_number, std::sync::atomic::Ordering::SeqCst);
        self.multi_spann_builder.invalidate(user_id, doc_id)
    }

    pub fn build(&mut self, base_directory: String, name: String) -> Result<()> {
        if self.finalized {
            return Err(anyhow::anyhow!("Cannot build a finalized segment"));
        }

        // Create necessary directories
        let segment_directory = format!("{base_directory}/{name}");
        std::fs::create_dir_all(&segment_directory)?;
        let term_directory = format!("{}/terms", segment_directory);
        std::fs::create_dir_all(&term_directory)?;

        // Build SPANN
        self.multi_spann_builder.build()?;
        let multi_spann_writer = MultiSpannWriter::new(segment_directory.clone());
        multi_spann_writer.write(&mut self.multi_spann_builder)?;

        // Build terms
        self.multi_term_builder.build()?;
        let multi_term_writer = MultiTermWriter::new_with_segment_dir(segment_directory.clone());
        multi_term_writer.write(&self.multi_term_builder)?;

        // Remove all reassigned_mappings file.
        self.remove_reassigned_mappings(&segment_directory)?;

        self.finalized = true;
        Ok(())
    }

    pub fn remove_reassigned_mappings(&self, base_directory: &str) -> Result<()> {
        let reassigned_mappings_files = std::fs::read_dir(base_directory)
            .expect("Failed to read directory")
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry
                    .file_name()
                    .to_str()
                    .unwrap()
                    .starts_with("reassigned_mappings.")
            })
            .collect::<Vec<_>>();
        for file in reassigned_mappings_files {
            std::fs::remove_file(file.path()).expect("Failed to remove file");
        }
        Ok(())
    }

    pub fn last_sequence_number(&self) -> u64 {
        self.last_sequence_number
            .load(std::sync::atomic::Ordering::SeqCst)
    }

    pub fn num_docs(&self) -> u64 {
        self.num_docs.load(std::sync::atomic::Ordering::Relaxed)
    }
}

unsafe impl Send for MutableSegment {}

unsafe impl Sync for MutableSegment {}

#[cfg(test)]
mod tests {
    use config::collection::CollectionConfig;

    use super::*;

    #[tokio::test]
    async fn test_mutable_segment() {
        let tmp_dir = tempdir::TempDir::new("mutable_segment_test").unwrap();
        let base_dir = tmp_dir.path().to_str().unwrap().to_string();

        let segment_config = CollectionConfig::default_test_config();
        let mutable_segment = MutableSegment::new(segment_config.clone(), base_dir)
            .expect("Failed to create mutable segment");

        assert!(mutable_segment.insert(0, &[1.0, 2.0, 3.0, 4.0]).is_ok());
        assert!(mutable_segment.insert(1, &[5.0, 6.0, 7.0, 8.0]).is_ok());
        assert!(mutable_segment.insert(2, &[9.0, 10.0, 11.0, 12.0]).is_ok());

        assert!(mutable_segment
            .invalidate(0, 0, 0)
            .expect("Failed to invalidate"));
        assert!(!mutable_segment
            .invalidate(0, 0, 1)
            .expect("Failed to invalidate"));
        assert!(mutable_segment.insert(0, &[5.0, 6.0, 7.0, 8.0]).is_ok());
        assert!(mutable_segment
            .invalidate(0, 0, 2)
            .expect("Failed to invalidate"));
    }

    #[tokio::test]
    async fn test_mutable_segment_build_with_terms() {
        let tmp_dir = tempdir::TempDir::new("mutable_segment_build_test").unwrap();
        let base_dir = tmp_dir.path().to_str().unwrap().to_string();

        let segment_config = CollectionConfig::default_test_config();
        let base_dir_for_segment = base_dir.clone();
        let mut mutable_segment = MutableSegment::new(segment_config.clone(), base_dir_for_segment)
            .expect("Failed to create mutable segment");

        // Insert a document with text attributes to ensure terms are generated
        let mut attributes = std::collections::HashMap::new();
        attributes.insert(
            "title".to_string(),
            proto::muopdb::AttributeValue {
                value: Some(proto::muopdb::attribute_value::Value::TextValue(
                    "test document title".to_string(),
                )),
            },
        );
        attributes.insert(
            "category".to_string(),
            proto::muopdb::AttributeValue {
                value: Some(proto::muopdb::attribute_value::Value::KeywordValue(
                    "test_category".to_string(),
                )),
            },
        );
        let doc_attr = proto::muopdb::DocumentAttribute { value: attributes };

        // Insert document with attributes
        assert!(mutable_segment
            .insert_for_user(0, 1, &[1.0, 2.0, 3.0, 4.0], 0, doc_attr)
            .is_ok());

        // Build the segment
        let segment_name = "test_segment";
        assert!(mutable_segment
            .build(base_dir.clone(), segment_name.to_string())
            .is_ok());

        // Check that term files are created
        let segment_dir = format!("{}/{}", base_dir, segment_name);
        let terms_dir = format!("{}/terms", segment_dir);

        // Verify the terms directory exists
        assert!(
            std::path::Path::new(&terms_dir).exists(),
            "Terms directory should exist"
        );

        // Verify the combined file exists (this is where TermWriter writes the term data)
        let combined_file_path = format!("{}/combined", terms_dir);
        assert!(
            std::path::Path::new(&combined_file_path).exists(),
            "Combined terms file should exist"
        );
    }

    #[tokio::test]
    async fn test_mutable_segment_build_with_terms_and_reindex() {
        let tmp_dir = tempdir::TempDir::new("mutable_segment_reindex_test").unwrap();
        let base_dir = tmp_dir.path().to_str().unwrap().to_string();

        // Create a config with reindex enabled
        let mut segment_config = CollectionConfig::default_test_config();
        segment_config.reindex = true;

        let base_dir_for_segment = base_dir.clone();
        let mut mutable_segment = MutableSegment::new(segment_config.clone(), base_dir_for_segment)
            .expect("Failed to create mutable segment");

        // Insert multiple documents with text attributes for multiple users
        // User 1 documents
        let mut attributes1 = std::collections::HashMap::new();
        attributes1.insert(
            "title".to_string(),
            proto::muopdb::AttributeValue {
                value: Some(proto::muopdb::attribute_value::Value::TextValue(
                    "apple banana cherry".to_string(),
                )),
            },
        );
        attributes1.insert(
            "category".to_string(),
            proto::muopdb::AttributeValue {
                value: Some(proto::muopdb::attribute_value::Value::KeywordValue(
                    "fruit".to_string(),
                )),
            },
        );
        let doc_attr1 = proto::muopdb::DocumentAttribute { value: attributes1 };

        let mut attributes2 = std::collections::HashMap::new();
        attributes2.insert(
            "title".to_string(),
            proto::muopdb::AttributeValue {
                value: Some(proto::muopdb::attribute_value::Value::TextValue(
                    "banana orange".to_string(),
                )),
            },
        );
        attributes2.insert(
            "category".to_string(),
            proto::muopdb::AttributeValue {
                value: Some(proto::muopdb::attribute_value::Value::KeywordValue(
                    "citrus".to_string(),
                )),
            },
        );
        let doc_attr2 = proto::muopdb::DocumentAttribute { value: attributes2 };

        // User 2 documents
        let mut attributes3 = std::collections::HashMap::new();
        attributes3.insert(
            "title".to_string(),
            proto::muopdb::AttributeValue {
                value: Some(proto::muopdb::attribute_value::Value::TextValue(
                    "dog cat mouse".to_string(),
                )),
            },
        );
        attributes3.insert(
            "category".to_string(),
            proto::muopdb::AttributeValue {
                value: Some(proto::muopdb::attribute_value::Value::KeywordValue(
                    "animal".to_string(),
                )),
            },
        );
        let doc_attr3 = proto::muopdb::DocumentAttribute { value: attributes3 };

        // Insert documents with attributes
        assert!(mutable_segment
            .insert_for_user(1, 1, &[1.0, 2.0, 3.0, 4.0], 0, doc_attr1)
            .is_ok());
        assert!(mutable_segment
            .insert_for_user(1, 2, &[5.0, 6.0, 7.0, 8.0], 1, doc_attr2)
            .is_ok());
        assert!(mutable_segment
            .insert_for_user(2, 1, &[9.0, 10.0, 11.0, 12.0], 0, doc_attr3)
            .is_ok());

        // Build the segment with reindex enabled
        let segment_name = "test_segment_reindex";
        assert!(mutable_segment
            .build(base_dir.clone(), segment_name.to_string())
            .is_ok());

        // Check that term files are created
        let segment_dir = format!("{}/{}", base_dir, segment_name);
        let terms_dir = format!("{}/terms", segment_dir);

        // Verify the terms directory exists
        assert!(
            std::path::Path::new(&terms_dir).exists(),
            "Terms directory should exist"
        );

        // Verify the combined file exists (this is where TermWriter writes the term data)
        let combined_file_path = format!("{}/combined", terms_dir);
        assert!(
            std::path::Path::new(&combined_file_path).exists(),
            "Combined terms file should exist"
        );

        // Since reindex is enabled, the MultiSpannWriter should have created reassigned mappings
        // that the MultiTermWriter reads and applies to the term indices

        // Load the multi-term index and verify terms are correctly indexed
        let multi_term_index = crate::multi_terms::index::MultiTermIndex::new(terms_dir.clone())
            .expect("Failed to load multi-term index");

        // Verify User 1's terms are accessible
        // Check for tokenized terms from text attributes
        assert!(multi_term_index
            .get_term_id_for_user(1, "title:appl")
            .is_ok());
        assert!(multi_term_index
            .get_term_id_for_user(1, "title:banana")
            .is_ok());
        assert!(multi_term_index
            .get_term_id_for_user(1, "title:cherri")
            .is_ok());
        assert!(multi_term_index
            .get_term_id_for_user(1, "title:orang")
            .is_ok());
        // Check for keyword terms
        assert!(multi_term_index
            .get_term_id_for_user(1, "category:fruit")
            .is_ok());
        assert!(multi_term_index
            .get_term_id_for_user(1, "category:citrus")
            .is_ok());

        // Verify User 2's terms are accessible
        assert!(multi_term_index
            .get_term_id_for_user(2, "title:dog")
            .is_ok());
        assert!(multi_term_index
            .get_term_id_for_user(2, "title:cat")
            .is_ok());
        assert!(multi_term_index
            .get_term_id_for_user(2, "title:mous")
            .is_ok());
        assert!(multi_term_index
            .get_term_id_for_user(2, "category:animal")
            .is_ok());

        // Verify posting lists contain the expected document IDs
        // For User 1, "title:banana" should appear in both documents (doc 1 and 2)
        let banana_id = multi_term_index
            .get_term_id_for_user(1, "title:banana")
            .unwrap();
        let banana_pl: Vec<u32> = multi_term_index
            .get_or_create_index(1)
            .unwrap()
            .get_posting_list_iterator(banana_id)
            .unwrap()
            .collect();
        assert_eq!(banana_pl.len(), 2, "Banana should appear in 2 documents");

        // For User 2, "title:dog" should appear in one document
        let dog_id = multi_term_index
            .get_term_id_for_user(2, "title:dog")
            .unwrap();
        let dog_pl: Vec<u32> = multi_term_index
            .get_or_create_index(2)
            .unwrap()
            .get_posting_list_iterator(dog_id)
            .unwrap()
            .collect();
        assert_eq!(dog_pl.len(), 1, "Dog should appear in 1 document");
    }

    #[tokio::test]
    async fn test_mutable_segment_french_stemming() {
        use config::attribute_schema::Language;
        let tmp_dir = tempdir::TempDir::new("mutable_segment_french_test").unwrap();
        let base_dir = tmp_dir.path().to_str().unwrap().to_string();

        let mut segment_config = CollectionConfig::default_test_config();
        let mut schema_map = std::collections::HashMap::new();
        schema_map.insert(
            "description".to_string(),
            AttributeType::Text(Language::French),
        );
        segment_config.attribute_schema = Some(AttributeSchema::new(schema_map));

        let mut mutable_segment = MutableSegment::new(segment_config.clone(), base_dir)
            .expect("Failed to create mutable segment");

        let mut attributes = std::collections::HashMap::new();
        attributes.insert(
            "description".to_string(),
            proto::muopdb::AttributeValue {
                value: Some(proto::muopdb::attribute_value::Value::TextValue(
                    "les chevaux".to_string(),
                )),
            },
        );
        let doc_attr = proto::muopdb::DocumentAttribute { value: attributes };

        mutable_segment
            .insert_for_user(1, 1, &[1.0, 2.0, 3.0, 4.0], 0, doc_attr)
            .unwrap();

        // Check if terms are stemmed: "description:le", "description:cheval"
        mutable_segment
            .build(
                tmp_dir.path().to_str().unwrap().to_string(),
                "french_seg".to_string(),
            )
            .unwrap();

        let terms_dir = format!("{}/french_seg/terms", tmp_dir.path().to_str().unwrap());
        let multi_term_index = crate::multi_terms::index::MultiTermIndex::new(terms_dir).unwrap();

        assert!(multi_term_index
            .get_term_id_for_user(1, "description:le")
            .is_ok());
        assert!(multi_term_index
            .get_term_id_for_user(1, "description:cheval")
            .is_ok());
    }
}
