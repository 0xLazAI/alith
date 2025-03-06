use crate::chunking::Chunker;
use std::path::PathBuf;

pub trait Knowledge: Chunker {
    /// Load the content into the memory.
    fn load(&self) -> Result<String, KnowledgeError>;
    /// Enrich the knowledge with the input string.
    fn enrich(&self, input: &str) -> Result<String, KnowledgeError>;
}

pub trait FileKnowledge: Knowledge {
    fn load_with_path(&self) -> Result<(PathBuf, String), KnowledgeError>;
}

#[derive(Debug, thiserror::Error)]
#[error("Knowledge error")]
pub enum KnowledgeError {
    #[error("Failed to load the knowledge source: {0}")]
    LoadError(String),
    #[error("An unknown error occurred: {0}")]
    Unknown(String),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}
