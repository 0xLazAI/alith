use alith_core::{
    chunking::{ChunkError, Chunker, chunk_text},
    knowledge::{Knowledge, KnowledgeError},
};
use anyhow::Result;

#[derive(Debug, Default)]
pub struct StringKnowledge {
    content: String,
}

impl StringKnowledge {
    pub fn new(content: impl ToString) -> Self {
        Self {
            content: content.to_string(),
        }
    }
}

impl Chunker for StringKnowledge {
    fn chunk(&self) -> std::result::Result<Vec<String>, ChunkError> {
        Ok(chunk_text(
            &self.content,
            self.chunk_size() as u32,
            self.overlap_percent(),
        )
        .map_err(|err| ChunkError::Normal(err.to_string()))?
        .unwrap_or_default())
    }
}

impl Knowledge for StringKnowledge {
    fn load(&self) -> Result<String, KnowledgeError> {
        Ok(self.content.clone())
    }

    fn enrich(&self, _input: &str) -> Result<String, KnowledgeError> {
        Ok(self.content.clone())
    }
}
