pub use llm_client::utils::chunking::{chunk_text, ChunkerConfig, ChunkerResult, TextChunker};

pub const DEFAULT_CHUNK_SIZE: usize = 1024;

pub trait Chunk: Send + Sync {
    fn chunk_size(&self) -> usize {
        DEFAULT_CHUNK_SIZE
    }

    fn overlap_percent(&self) -> Option<f32> {
        None
    }

    fn chunk(&self) -> Result<Vec<String>, ChunkError>;
}

/// An enumeration of possible errors that may occur during chunk operations.
#[derive(Debug, thiserror::Error)]
pub enum ChunkError {
    /// A generic chunk error.
    #[error("A normal chunk error occurred: {0}")]
    Normal(String),
}
