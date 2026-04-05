use unicode_segmentation::{UnicodeSegmentation, UnicodeWordIndices};

use super::{Token, TokenStream, Tokenizer};

/// A tokenizer that uses Unicode word boundary detection.
///
/// This tokenizer splits text into words using proper Unicode word boundary
/// detection (per Unicode Standard Annex #29). It handles punctuation,
/// numbers, and various scripts correctly.
pub struct UnicodeTokenizer;

impl Default for UnicodeTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl UnicodeTokenizer {
    /// Creates a new `UnicodeTokenizer`.
    pub fn new() -> Self {
        Self
    }
}

impl Tokenizer for UnicodeTokenizer {
    type TokenStream<'a> = UnicodeTokenStream<'a>;

    fn input<'a>(&self, text: &'a str) -> Self::TokenStream<'a> {
        UnicodeTokenStream {
            words: text.unicode_word_indices(),
            current_token: Token::new(String::new()),
        }
    }
}

/// Token stream that yields words based on Unicode word boundaries.
pub struct UnicodeTokenStream<'a> {
    words: UnicodeWordIndices<'a>,
    current_token: Token,
}

impl<'a> TokenStream for UnicodeTokenStream<'a> {
    fn advance(&mut self) -> bool {
        if let Some((_, word)) = self.words.next() {
            self.current_token = Token::new(word.to_lowercase());
            true
        } else {
            false
        }
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
    use super::*;

    #[test]
    fn test_unicode_tokenizer_basic() {
        let tokenizer = UnicodeTokenizer::new();
        let mut stream = tokenizer.input("hello world");

        let mut tokens = Vec::new();
        while let Some(token) = stream.next() {
            tokens.push(token.text);
        }

        assert_eq!(tokens, vec!["hello", "world"]);
    }

    #[test]
    fn test_unicode_tokenizer_punctuation() {
        let tokenizer = UnicodeTokenizer::new();
        let mut stream = tokenizer.input("Hello, world! How are you?");

        let mut tokens = Vec::new();
        while let Some(token) = stream.next() {
            tokens.push(token.text);
        }

        // Punctuation should be removed, only words remain
        assert_eq!(tokens, vec!["hello", "world", "how", "are", "you"]);
    }

    #[test]
    fn test_unicode_tokenizer_empty() {
        let tokenizer = UnicodeTokenizer::new();
        let mut stream = tokenizer.input("");

        assert!(stream.next().is_none());
    }

    #[test]
    fn test_unicode_tokenizer_whitespace_only() {
        let tokenizer = UnicodeTokenizer::new();
        let mut stream = tokenizer.input("   ");

        assert!(stream.next().is_none());
    }

    #[test]
    fn test_unicode_tokenizer_numbers() {
        let tokenizer = UnicodeTokenizer::new();
        let mut stream = tokenizer.input("Version 1.2.3 release");

        let mut tokens = Vec::new();
        while let Some(token) = stream.next() {
            tokens.push(token.text);
        }

        // "1.2.3" is treated as a single word by Unicode word boundaries
        assert_eq!(tokens, vec!["version", "1.2.3", "release"]);
    }

    #[test]
    fn test_unicode_tokenizer_lowercase() {
        let tokenizer = UnicodeTokenizer::new();
        let mut stream = tokenizer.input("HELLO World");

        let mut tokens = Vec::new();
        while let Some(token) = stream.next() {
            tokens.push(token.text);
        }

        // All tokens should be lowercased
        assert_eq!(tokens, vec!["hello", "world"]);
    }

    #[test]
    fn test_unicode_tokenizer_mixed_symbols() {
        let tokenizer = UnicodeTokenizer::new();
        let mut stream = tokenizer.input("email@example.com");

        let mut tokens = Vec::new();
        while let Some(token) = stream.next() {
            tokens.push(token.text);
        }

        // "example.com" is treated as a single word, "email" is separate before @
        assert_eq!(tokens, vec!["email", "example.com"]);
    }
}
