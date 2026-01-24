use rust_stemmers::{Algorithm, Stemmer};

use super::{Token, TokenStream, Tokenizer};
use crate::tokenizer::white_space_tokenizer::WhiteSpaceTokenStream;

/// A tokenizer that stems words using the Snowball algorithm.
///
/// This tokenizer first splits the input on whitespace, then lowercases
/// each token and reduces it to its root/stem form.
pub struct StemmingTokenizer {
    algorithm: Algorithm,
}

use config::attribute_schema::Language;

impl StemmingTokenizer {
    /// Creates a new `StemmingTokenizer` for the specified language.
    ///
    /// # Arguments
    /// * `lang` - The language to use for stemming.
    pub fn for_language(lang: Language) -> Self {
        let algorithm = match lang {
            Language::Arabic => Algorithm::Arabic,
            Language::Danish => Algorithm::Danish,
            Language::Dutch => Algorithm::Dutch,
            Language::English | Language::Vietnamese => Algorithm::English,
            Language::Finnish => Algorithm::Finnish,
            Language::French => Algorithm::French,
            Language::German => Algorithm::German,
            Language::Greek => Algorithm::Greek,
            Language::Hungarian => Algorithm::Hungarian,
            Language::Italian => Algorithm::Italian,
            Language::Norwegian => Algorithm::Norwegian,
            Language::Portuguese => Algorithm::Portuguese,
            Language::Romanian => Algorithm::Romanian,
            Language::Russian => Algorithm::Russian,
            Language::Spanish => Algorithm::Spanish,
            Language::Swedish => Algorithm::Swedish,
            Language::Tamil => Algorithm::Tamil,
            Language::Turkish => Algorithm::Turkish,
        };
        Self::new(algorithm)
    }

    /// Creates a new `StemmingTokenizer` with the specified stemming algorithm.
    ///
    /// # Arguments
    /// * `algorithm` - The Snowball stemming algorithm to use.
    pub fn new(algorithm: Algorithm) -> Self {
        Self { algorithm }
    }
}

impl Default for StemmingTokenizer {
    fn default() -> Self {
        Self::new(Algorithm::English)
    }
}

impl Tokenizer for StemmingTokenizer {
    type TokenStream<'a> = StemmingTokenStream<'a>;

    fn input<'a>(&self, text: &'a str) -> StemmingTokenStream<'a> {
        StemmingTokenStream::new(text, self.algorithm)
    }
}

pub struct StemmingTokenStream<'a> {
    inner: WhiteSpaceTokenStream<'a>,
    stemmer: Stemmer,
    token: Token,
}

impl<'a> StemmingTokenStream<'a> {
    /// Creates a new `StemmingTokenStream`.
    pub fn new(text: &'a str, algorithm: Algorithm) -> Self {
        Self {
            inner: WhiteSpaceTokenStream::new(text),
            stemmer: Stemmer::create(algorithm),
            token: Token::new(String::new()),
        }
    }
}

impl<'a> TokenStream for StemmingTokenStream<'a> {
    fn advance(&mut self) -> bool {
        if self.inner.advance() {
            let original_token = self.inner.token();
            // Lowercase and stem
            let lowercased = original_token.text.to_lowercase();
            let stemmed = self.stemmer.stem(&lowercased).into_owned();
            self.token = Token::new(stemmed);
            true
        } else {
            false
        }
    }

    fn token(&self) -> Token {
        self.token.clone()
    }

    fn next(&mut self) -> Option<Token> {
        if self.advance() {
            Some(self.token.clone())
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use rust_stemmers::Algorithm;

    use super::*;

    #[test]
    fn test_stemming_tokenizer_basic() {
        let tokenizer = StemmingTokenizer::new(Algorithm::English);
        let mut stream = tokenizer.input("running connections happily");

        let mut tokens = Vec::new();
        while let Some(token) = stream.next() {
            tokens.push(token.text);
        }

        assert_eq!(tokens, vec!["run", "connect", "happili"]);
    }

    #[test]
    fn test_stemming_tokenizer_whitespace() {
        let tokenizer = StemmingTokenizer::new(Algorithm::English);
        let mut stream = tokenizer.input("  fast   faster    fastest  ");

        let mut tokens = Vec::new();
        while let Some(token) = stream.next() {
            tokens.push(token.text);
        }

        assert_eq!(tokens, vec!["fast", "faster", "fastest"]);
    }

    #[test]
    fn test_stemming_tokenizer_case_insensitivity() {
        let tokenizer = StemmingTokenizer::new(Algorithm::English);
        let mut stream = tokenizer.input("Running CONNECTIONS");

        let mut tokens = Vec::new();
        while let Some(token) = stream.next() {
            tokens.push(token.text);
        }

        assert_eq!(tokens, vec!["run", "connect"]);
    }

    #[test]
    fn test_stemming_tokenizer_empty() {
        let tokenizer = StemmingTokenizer::default();
        let mut stream = tokenizer.input("   ");
        assert!(stream.next().is_none());
    }

    #[test]
    fn test_stemming_tokenizer_french() {
        let tokenizer = StemmingTokenizer::for_language(Language::French);
        let mut stream = tokenizer.input("les chevaux courent");

        let mut tokens = Vec::new();
        while let Some(token) = stream.next() {
            tokens.push(token.text);
        }

        // French: les -> le, chevaux -> cheval, courent -> courent
        assert_eq!(tokens, vec!["le", "cheval", "courent"]);
    }

    #[test]
    fn test_stemming_tokenizer_vietnamese() {
        let tokenizer = StemmingTokenizer::for_language(Language::Vietnamese);
        let mut stream = tokenizer.input("running tests");

        let mut tokens = Vec::new();
        while let Some(token) = stream.next() {
            tokens.push(token.text);
        }

        // Vietnamese should use English stemmer: running -> run, tests -> test
        assert_eq!(tokens, vec!["run", "test"]);
    }
}
