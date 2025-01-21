pub mod html;
pub mod pdf;
pub mod string;
pub mod text;

pub use alith_core::{
    chunking::{chunk_text, Chunk, ChunkError},
    knowledge::{FileKnowledge, Knowledge, KnowledgeError},
};
