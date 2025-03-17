use std::fs::read_to_string;
use std::path::{Path, PathBuf};

use alith_core::{
    chunking::{ChunkError, Chunker, chunk_text},
    knowledge::{FileKnowledge, Knowledge, KnowledgeError},
};

pub struct TextFileKnowledge {
    pub path: PathBuf,
}

impl TextFileKnowledge {
    pub fn new<P: AsRef<Path>>(path: P) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
        }
    }
}

impl Chunker for TextFileKnowledge {
    fn chunk(&self) -> std::result::Result<Vec<String>, ChunkError> {
        Ok(chunk_text(
            &self
                .load()
                .map_err(|err| ChunkError::Normal(err.to_string()))?,
            self.chunk_size() as u32,
            self.overlap_percent(),
        )
        .map_err(|err| ChunkError::Normal(err.to_string()))?
        .unwrap_or_default())
    }
}

impl Knowledge for TextFileKnowledge {
    fn load(&self) -> Result<String, KnowledgeError> {
        Ok(read_to_string(&self.path)?)
    }

    fn enrich(&self, _input: &str) -> Result<String, KnowledgeError> {
        Ok(format!("<textfile>{}</textfile>", self.load()?))
    }
}

impl FileKnowledge for TextFileKnowledge {
    fn load_with_path(&self) -> Result<(PathBuf, String), KnowledgeError> {
        let content = self.load()?;
        Ok((self.path.clone(), content))
    }
}
