pub mod stemming_tokenizer;
pub mod stop_word_filter;
pub mod unicode_tokenizer;
pub mod white_space_tokenizer;

use std::collections::HashSet;

use config::attribute_schema::Language;
use stemming_tokenizer::StemmingTokenizer;
use unicode_tokenizer::UnicodeTokenizer;

#[derive(Clone)]
pub struct Token {
    pub text: String,
}

impl Token {
    /// Creates a new `Token` with the given text.
    ///
    /// # Arguments
    /// * `text` - The string content of the token.
    ///
    /// # Returns
    /// * `Self` - A new token instance.
    pub fn new(text: String) -> Self {
        Self { text }
    }
}

pub trait TokenStream {
    /// Advances the stream to the next token.
    ///
    /// # Returns
    /// * `bool` - `true` if a token was successfully loaded, `false` if the end of the stream was reached.
    fn advance(&mut self) -> bool;

    /// Returns the current token in the stream.
    ///
    /// This should only be called after a successful call to `advance`.
    ///
    /// # Returns
    /// * `Token` - The current token instance.
    fn token(&self) -> Token;

    /// Advances the stream and returns the next token.
    ///
    /// # Returns
    /// * `Option<Token>` - The next token if available, otherwise `None`.
    fn next(&mut self) -> Option<Token>;
}

pub trait Tokenizer {
    type TokenStream<'a>: TokenStream;

    /// Creates a token stream for the given input text.
    ///
    /// # Arguments
    /// * `text` - The input string to be tokenized.
    ///
    /// # Returns
    /// * `Self::TokenStream<'a>` - A stream of tokens extracted from the input text.
    fn input<'a>(&self, text: &'a str) -> Self::TokenStream<'a>;
}

/// Complete tokenization pipeline: Unicode segmentation -> stemming -> stop word filtering.
///
/// This pipeline combines three stages:
/// 1. Unicode word boundary segmentation (splits text into words, handles punctuation)
/// 2. Stemming (reduces words to their root form using language-specific algorithms)
/// 3. Stop word filtering (removes common words that don't contribute to meaning)
///
/// # Example
/// ```
/// use index::tokenizer::TokenizerPipeline;
///
/// let pipeline = TokenizerPipeline::english();
/// let tokens = pipeline.tokenize("The quick brown fox");
/// // Returns: ["quick", "brown", "fox"] - "the" is filtered as a stop word
/// ```
pub struct TokenizerPipeline {
    unicode_tokenizer: UnicodeTokenizer,
    stemmer: StemmingTokenizer,
    stop_words: HashSet<String>,
}

impl TokenizerPipeline {
    /// Create a new pipeline with English defaults.
    ///
    /// Uses English stemming and a standard set of English stop words.
    pub fn english() -> Self {
        Self {
            unicode_tokenizer: UnicodeTokenizer::new(),
            stemmer: StemmingTokenizer::for_language(Language::English),
            stop_words: Self::english_stop_words(),
        }
    }

    /// Create with custom language and no stop words.
    ///
    /// # Arguments
    /// * `lang` - The language to use for stemming
    pub fn with_language(lang: Language) -> Self {
        Self {
            unicode_tokenizer: UnicodeTokenizer::new(),
            stemmer: StemmingTokenizer::for_language(lang),
            stop_words: HashSet::new(),
        }
    }

    /// Create a new pipeline with custom stop words.
    ///
    /// # Arguments
    /// * `stop_words` - Set of words to filter out
    pub fn with_stop_words(stop_words: HashSet<String>) -> Self {
        Self {
            unicode_tokenizer: UnicodeTokenizer::new(),
            stemmer: StemmingTokenizer::for_language(Language::English),
            stop_words,
        }
    }

    /// Tokenize text into final stemmed, filtered tokens.
    ///
    /// # Arguments
    /// * `text` - The input text to tokenize
    ///
    /// # Returns
    /// * `Vec<String>` - List of processed tokens
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        let mut tokens = Vec::new();

        // Step 1: Unicode word segmentation
        let mut unicode_stream = self.unicode_tokenizer.input(text);
        while let Some(token) = unicode_stream.next() {
            // Step 2: Stemming (re-tokenize each word through stemmer)
            let mut stem_stream = self.stemmer.input(&token.text);
            if let Some(stemmed) = stem_stream.next() {
                // Step 3: Stop word filtering
                if !self.stop_words.contains(&stemmed.text) {
                    tokens.push(stemmed.text);
                }
            }
        }
        tokens
    }

    fn english_stop_words() -> HashSet<String> {
        let mut set = HashSet::new();
        for word in &[
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", "in",
            "is", "it", "its", "of", "on", "that", "the", "to", "was", "will", "with",
        ] {
            set.insert(word.to_string());
        }
        set
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_english() {
        let pipeline = TokenizerPipeline::english();
        let tokens = pipeline.tokenize("The running dogs are happy");

        // "the", "are" filtered as stop words
        // "running" -> "run", "dogs" -> "dog", "happy" -> "happi"
        assert_eq!(tokens, vec!["run", "dog", "happi"]);
    }

    #[test]
    fn test_pipeline_french() {
        let pipeline = TokenizerPipeline::with_language(Language::French);
        let tokens = pipeline.tokenize("les chevaux courent");

        // "les" -> "le" (stemmed), but with empty stop words set, all pass through
        // "chevaux" -> "cheval", "courent" remains "courent"
        assert_eq!(tokens, vec!["le", "cheval", "courent"]);
    }

    #[test]
    fn test_pipeline_empty() {
        let pipeline = TokenizerPipeline::english();
        let tokens = pipeline.tokenize("");

        assert!(tokens.is_empty());
    }

    #[test]
    fn test_pipeline_only_stop_words() {
        let pipeline = TokenizerPipeline::english();
        let tokens = pipeline.tokenize("the and of a an");

        assert!(tokens.is_empty());
    }

    #[test]
    fn test_pipeline_unicode() {
        let pipeline = TokenizerPipeline::english();
        let tokens = pipeline.tokenize("Hello, world! Testing...");

        // Punctuation stripped, "hello" -> "hello" (already stem), "world" -> "world", "testing" -> "test"
        assert_eq!(tokens, vec!["hello", "world", "test"]);
    }
}

// Re-export types from submodules
pub use stemming_tokenizer::StemmingTokenStream;
pub use stop_word_filter::StopWordFilter;
pub use unicode_tokenizer::UnicodeTokenStream;
pub use white_space_tokenizer::WhiteSpaceTokenStream;
