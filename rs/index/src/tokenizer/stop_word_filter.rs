use std::collections::HashSet;

use super::{Token, TokenStream};

/// TokenStream wrapper that filters out stop words.
///
/// This filter wraps any TokenStream and removes tokens that are present
/// in the configured stop word set. It provides a `with_english_defaults()`
/// convenience constructor with common English stop words.
pub struct StopWordFilter<T: TokenStream> {
    inner: T,
    stop_words: HashSet<String>,
    current_token: Token,
}

impl<T: TokenStream> StopWordFilter<T> {
    /// Create a new StopWordFilter wrapping an inner TokenStream.
    ///
    /// # Arguments
    /// * `inner` - The TokenStream to wrap
    /// * `stop_words` - Set of stop words to filter out
    pub fn new(inner: T, stop_words: HashSet<String>) -> Self {
        Self {
            inner,
            stop_words,
            current_token: Token::new(String::new()),
        }
    }

    /// Create with default English stop words.
    pub fn with_english_defaults(inner: T) -> Self {
        let mut stop_words = HashSet::new();
        // Common English stop words
        for word in &[
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", "in",
            "is", "it", "its", "of", "on", "that", "the", "to", "was", "will", "with",
        ] {
            stop_words.insert(word.to_string());
        }
        Self::new(inner, stop_words)
    }

    fn advance_skipping_stop_words(&mut self) -> bool {
        while self.inner.advance() {
            let token = self.inner.token();
            if !self.stop_words.contains(&token.text) {
                self.current_token = token;
                return true;
            }
        }
        false
    }
}

impl<T: TokenStream> TokenStream for StopWordFilter<T> {
    fn advance(&mut self) -> bool {
        self.advance_skipping_stop_words()
    }

    fn token(&self) -> Token {
        self.current_token.clone()
    }

    fn next(&mut self) -> Option<Token> {
        if self.advance() {
            Some(self.current_token.clone())
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use crate::tokenizer::stop_word_filter::StopWordFilter;
    use crate::tokenizer::white_space_tokenizer::WhiteSpaceTokenizer;
    use crate::tokenizer::{TokenStream, Tokenizer};

    #[test]
    fn test_stop_word_filter_removes_stops() {
        let tokenizer = WhiteSpaceTokenizer {};
        let stream = tokenizer.input("the quick brown fox");
        let mut filter = StopWordFilter::with_english_defaults(stream);

        let mut tokens = Vec::new();
        while let Some(token) = filter.next() {
            tokens.push(token.text);
        }

        // "the" should be removed, others should remain
        assert_eq!(tokens, vec!["quick", "brown", "fox"]);
    }

    #[test]
    fn test_stop_word_filter_passes_non_stops() {
        let tokenizer = WhiteSpaceTokenizer {};
        let stream = tokenizer.input("hello world");
        let mut filter = StopWordFilter::with_english_defaults(stream);

        let mut tokens = Vec::new();
        while let Some(token) = filter.next() {
            tokens.push(token.text);
        }

        // Both should pass through
        assert_eq!(tokens, vec!["hello", "world"]);
    }

    #[test]
    fn test_stop_word_filter_all_stops() {
        let tokenizer = WhiteSpaceTokenizer {};
        let stream = tokenizer.input("the and of");
        let mut filter = StopWordFilter::with_english_defaults(stream);

        let mut tokens = Vec::new();
        while let Some(token) = filter.next() {
            tokens.push(token.text);
        }

        // All are stop words, should return empty
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_stop_word_filter_custom() {
        let tokenizer = WhiteSpaceTokenizer {};
        let stream = tokenizer.input("foo hello bar world");

        let mut custom_stops = HashSet::new();
        custom_stops.insert("foo".to_string());
        custom_stops.insert("bar".to_string());

        let mut filter = StopWordFilter::new(stream, custom_stops);

        let mut tokens = Vec::new();
        while let Some(token) = filter.next() {
            tokens.push(token.text);
        }

        // Only custom stops "foo" and "bar" should be removed
        assert_eq!(tokens, vec!["hello", "world"]);
    }

    #[test]
    fn test_stop_word_filter_empty_input() {
        let tokenizer = WhiteSpaceTokenizer {};
        let stream = tokenizer.input("");
        let mut filter = StopWordFilter::with_english_defaults(stream);

        let mut tokens = Vec::new();
        while let Some(token) = filter.next() {
            tokens.push(token.text);
        }

        assert!(tokens.is_empty());
    }
}
