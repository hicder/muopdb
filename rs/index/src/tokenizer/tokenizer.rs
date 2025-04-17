
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
}

/*pub trait Tokenizer: TokenStream {
    fn tokenstream(text: String) -> TokenStream;
*/

mod tests {
    
}
