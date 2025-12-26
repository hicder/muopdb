use std::str::CharIndices;

use super::tokenizer::{Token, TokenStream, Tokenizer};

pub struct WhiteSpaceTokenizer {}

impl Tokenizer for WhiteSpaceTokenizer {
    type TokenStream<'a> = WhiteSpaceTokenStream<'a>;

    /// Creates a new `WhiteSpaceTokenStream` for the given input text.
    ///
    /// # Arguments
    /// * `text` - The input string to be tokenized by whitespace.
    ///
    /// # Returns
    /// * `WhiteSpaceTokenStream<'a>` - A stream that splits the input into tokens based on whitespace.
    fn input<'a>(&self, text: &'a str) -> WhiteSpaceTokenStream<'a> {
        WhiteSpaceTokenStream::new(text)
    }
}

pub struct WhiteSpaceTokenStream<'a> {
    text: &'a str,
    chars: CharIndices<'a>,
    token: Token,
}

impl<'a> WhiteSpaceTokenStream<'a> {
    /// Creates a new `WhiteSpaceTokenStream` from the provided text.
    ///
    /// # Arguments
    /// * `text` - The string to be tokenized.
    ///
    /// # Returns
    /// * `Self` - A new token stream instance.
    pub fn new(text: &'a str) -> Self {
        WhiteSpaceTokenStream {
            text,
            chars: text.char_indices(),
            token: Token::new(String::from("")),
        }
    }

    /// Skips leading whitespace characters and returns the byte offset of the next token's start.
    ///
    /// # Returns
    /// * `Option<usize>` - The byte offset of the token start, or `None` if the end of the text is reached.
    fn find_start(&mut self) -> Option<usize> {
        loop {
            let next_value = self.chars.next();
            match next_value {
                Some(ch) => {
                    if ch.1.is_whitespace() {
                        continue;
                    } else {
                        return Some(ch.0);
                    }
                }
                None => {
                    return None;
                }
            }
        }
    }

    /// Traverses non-whitespace characters and returns the byte offset of the next whitespace or the end of the text.
    ///
    /// # Returns
    /// * `usize` - The byte offset where the current token ends.
    fn find_end(&mut self) -> usize {
        loop {
            let next_value = self.chars.next();
            match next_value {
                Some(ch) => {
                    if ch.1.is_whitespace() {
                        return ch.0;
                    }
                }
                None => {
                    return self.text.len();
                }
            }
        }
    }
}

impl<'a> TokenStream for WhiteSpaceTokenStream<'a> {
    /// Advances the stream to the next whitespace-separated token.
    ///
    /// # Returns
    /// * `bool` - `true` if a token was successfully found, `false` otherwise.
    fn advance(&mut self) -> bool {
        if let Some(start_offset) = self.find_start() {
            let end_offset = self.find_end();
            self.token = Token::new(String::from(&self.text[start_offset..end_offset]));
            true
        } else {
            false
        }
    }

    /// Returns the most recently extracted token.
    ///
    /// # Returns
    /// * `Token` - The current token instance.
    fn token(&self) -> Token {
        self.token.clone()
    }

    /// Advances the stream and returns the next token if available.
    ///
    /// # Returns
    /// * `Option<Token>` - The next token in the stream, or `None` if end-of-input is reached.
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
    use crate::tokenizer::tokenizer::{TokenStream, Tokenizer};
    use crate::tokenizer::white_space_tokenizer::{WhiteSpaceTokenStream, WhiteSpaceTokenizer};

    #[test]
    fn test_token_stream_simple() {
        let mut token_stream = WhiteSpaceTokenStream::new("happy new year");
        let mut tokens: Vec<String> = vec![];
        while token_stream.advance() {
            tokens.push(token_stream.token().text);
        }

        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0], "happy");
        assert_eq!(tokens[1], "new");
        assert_eq!(tokens[2], "year");
    }

    #[test]
    fn test_tokenizer() {
        let tokenizer = WhiteSpaceTokenizer {};
        let mut token_stream: WhiteSpaceTokenStream<'_> =
            tokenizer.input("   .  happy      new  year ");
        let mut tokens: Vec<String> = vec![];
        while let Some(token) = token_stream.next() {
            tokens.push(token.text.clone());
        }

        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens[0], ".");
        assert_eq!(tokens[1], "happy");
        assert_eq!(tokens[2], "new");
        assert_eq!(tokens[3], "year");
    }
}
