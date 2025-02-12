use std::path::Path;

pub use tokenizers::Tokenizer;

#[derive(Debug, thiserror::Error)]
#[error("Tokenizer error")]
pub enum TokenizerError {
    #[error("An unknown error occurred: {0}")]
    Unknown(String),
    #[error("Failed to load the tokenizer: {0}")]
    LoadError(String),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Load tokenizer from the file path.
#[inline]
pub fn load_tokenizer<P: AsRef<Path>>(file: P) -> Result<Tokenizer, TokenizerError> {
    Tokenizer::from_file(file).map_err(|err| TokenizerError::LoadError(err.to_string()))
}
