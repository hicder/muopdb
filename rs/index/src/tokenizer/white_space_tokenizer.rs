use std::str::CharIndices;

use super::tokenizer::{TokenStream, Token};

pub struct WhiteSpaceTokenizer<'a>{
    text: &'a str,
    chars: CharIndices<'a>,
    token: Token,
}

impl<'a> WhiteSpaceTokenizer<'a> {
    pub fn new(text: &'a str) -> Self {
        WhiteSpaceTokenizer{
            text, 
            chars: text.char_indices(), 
            token: Token::new(String::from("")),
        }
    }

    fn find_start(&mut self) -> Option<usize> {
        loop {
            let next_value = self.chars.next();
            match next_value {
                Some(ch) => {
                    if ch.1.is_whitespace() == true {
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

    fn find_end(&mut self) -> usize {
        loop {
            let next_value = self.chars.next();
            match next_value {
                Some(ch) => {
                    if ch.1.is_whitespace() == true {
                        return ch.0;
                    }
                },
                None => {
                    return self.text.len();
                },
            }
        }
    }
}
 
impl<'a> TokenStream for WhiteSpaceTokenizer<'a> {
    fn advance(&mut self) -> bool {
        if let Some(start_offset) = self.find_start() {
            let end_offset = self.find_end();
            self.token = Token::new(String::from(&self.text[start_offset..end_offset]));
            return true;
        } else {
            return false;
        }
    }

    fn token(&self) -> Token {
        self.token.clone()
    }
}

#[cfg(test)]
mod tests {
    use crate::tokenizer::tokenizer::TokenStream;
    use crate::tokenizer::white_space_tokenizer::WhiteSpaceTokenizer;

    #[test]
    fn test_token_stream_simple() {
        let mut tokenizer = WhiteSpaceTokenizer::new("happy new year");
        let mut tokens: Vec<String> = vec![];
        while tokenizer.advance() == true {
            tokens.push(tokenizer.token().text);
        }

        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0], "happy");
        assert_eq!(tokens[1], "new");
        assert_eq!(tokens[2], "year");
    }

    #[test]
    fn test_token_stream_multiple_whitespace() {
        let mut tokenizer = WhiteSpaceTokenizer::new("   .  happy      new  year ");
        let mut tokens: Vec<String> = vec![];
        while tokenizer.advance() == true {
            tokens.push(tokenizer.token().text);
        }

        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens[0], ".");
        assert_eq!(tokens[1], "happy");
        assert_eq!(tokens[2], "new");
        assert_eq!(tokens[3], "year");
    }
}
