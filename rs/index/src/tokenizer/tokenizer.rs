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
