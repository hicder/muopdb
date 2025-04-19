#[derive(Clone)]
pub struct Token {
    pub text: String,
}

impl Token {
    pub fn new(text: String) -> Self {
        Self { text }
    }
}

pub trait TokenStream {
    fn advance(&mut self) -> bool;

    fn token(&self) -> Token;

    fn next(&mut self) -> Option<Token>;
}

pub trait Tokenizer {
    type TokenStream<'a>: TokenStream;

    fn input<'a>(&mut self, text: &'a str) -> Self::TokenStream<'a>;
}
