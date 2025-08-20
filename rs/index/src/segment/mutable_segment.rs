use std::sync::atomic::AtomicU64;
use std::sync::Mutex;
use std::time::Instant;

use anyhow::{Ok, Result};
use config::collection::CollectionConfig;
use log::debug;
use proto::muopdb::DocumentAttribute;

use crate::multi_spann::builder::MultiSpannBuilder;
use crate::multi_spann::writer::MultiSpannWriter;
use crate::terms::builder::TermBuilder;
use crate::terms::writer::TermWriter;
use crate::tokenizer::tokenizer::{TokenStream, Tokenizer};
use crate::tokenizer::white_space_tokenizer::WhiteSpaceTokenizer;

pub struct MutableSegment {
    multi_spann_builder: MultiSpannBuilder,
    term_builder: Mutex<TermBuilder>,

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
            multi_spann_builder: MultiSpannBuilder::new(config, base_directory)?,
            term_builder: Mutex::new(TermBuilder::new(&term_directory)),
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
        self.insert_for_user(0, doc_id, data, 0, None)
    }

    /// Insert a document for a user
    pub fn insert_for_user(
        &self,
        user_id: u128,
        doc_id: u128,
        data: &[f32],
        sequence_number: u64,
        document_attribute: Option<DocumentAttribute>,
    ) -> Result<()> {
        debug!(
            "Inserting for user: {user_id}, doc_id: {doc_id}, sequence_number: {sequence_number}"
        );
        if self.finalized {
            return Err(anyhow::anyhow!("Cannot insert into a finalized segment"));
        }

        self.multi_spann_builder.insert(user_id, doc_id, data)?;

        // Process document attributes if present
        if let Some(attributes) = document_attribute {
            let mut tokenizer = WhiteSpaceTokenizer {};
            let mut term_builder = self.term_builder.lock().unwrap();
            let doc_id_u64 = doc_id as u64;

            for (attr_name, attr_value) in attributes.value {
                match attr_value.value {
                    Some(proto::muopdb::attribute_value::Value::TextValue(text)) => {
                        // Tokenize the text attribute
                        let mut token_stream = tokenizer.input(&text);

                        // Process each token and insert with the term builder
                        while let Some(token) = token_stream.next() {
                            let term = format!("{}:{}", attr_name, token.text);
                            term_builder.add(doc_id_u64, term)?;
                        }
                    }
                    Some(proto::muopdb::attribute_value::Value::KeywordValue(keyword)) => {
                        // For keyword attributes, insert the whole keyword as a single term
                        let term = format!("{}:{}", attr_name, keyword);
                        term_builder.add(doc_id_u64, term)?;
                    }
                    _ => {
                        // Other attribute types (int, float, bool) are not tokenized
                        continue;
                    }
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

        let segment_directory = format!("{base_directory}/{name}");
        std::fs::create_dir_all(&segment_directory)?;

        self.multi_spann_builder.build()?;

        // Build the term builder
        self.term_builder.lock().unwrap().build()?;

        // Write the term builder using term writer
        let term_directory = format!("{}/terms", segment_directory);
        std::fs::create_dir_all(&term_directory)?;
        let term_writer = TermWriter::new(term_directory);
        term_writer.write(&mut self.term_builder.lock().unwrap())?;

        let multi_spann_writer = MultiSpannWriter::new(segment_directory);
        multi_spann_writer.write(&mut self.multi_spann_builder)?;
        self.finalized = true;
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
            .insert_for_user(0, 1, &[1.0, 2.0, 3.0, 4.0], 0, Some(doc_attr))
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

        // Verify the term_map file exists (temporary file used during building)
        let term_map_path = format!("{}/term_map", terms_dir);
        assert!(
            std::path::Path::new(&term_map_path).exists(),
            "Term map file should exist"
        );

        // Verify the posting_lists file exists (temporary file used during building)
        let posting_lists_path = format!("{}/posting_lists", terms_dir);
        assert!(
            std::path::Path::new(&posting_lists_path).exists(),
            "Posting lists file should exist"
        );

        // Verify the offsets file exists (temporary file used during building)
        let offsets_path = format!("{}/offsets", terms_dir);
        assert!(
            std::path::Path::new(&offsets_path).exists(),
            "Offsets file should exist"
        );
    }
}
