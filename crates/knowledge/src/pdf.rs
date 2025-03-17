use lopdf::Document;
use std::path::{Path, PathBuf};

use alith_core::{
    chunking::{ChunkError, Chunker, chunk_text},
    knowledge::{FileKnowledge, Knowledge, KnowledgeError},
};

pub struct PdfFileKnowledge {
    pub path: PathBuf,
}

impl PdfFileKnowledge {
    pub fn new<P: AsRef<Path>>(path: P) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
        }
    }
}

impl Chunker for PdfFileKnowledge {
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

impl Knowledge for PdfFileKnowledge {
    fn load(&self) -> Result<String, KnowledgeError> {
        let doc =
            Document::load(&self.path).map_err(|err| KnowledgeError::LoadError(err.to_string()))?;
        Ok(doc
            .page_iter()
            .enumerate()
            .map(|(page_no, _)| {
                doc.extract_text(&[page_no as u32 + 1])
                    .map_err(|err| KnowledgeError::LoadError(err.to_string()))
            })
            .collect::<Result<Vec<String>, KnowledgeError>>()?
            .into_iter()
            .collect::<String>())
    }

    fn enrich(&self, _input: &str) -> Result<String, KnowledgeError> {
        Ok(format!("<pdffile>{}</pdffile>", self.load()?))
    }
}

impl FileKnowledge for PdfFileKnowledge {
    fn load_with_path(&self) -> Result<(PathBuf, String), KnowledgeError> {
        let content = self.load()?;
        Ok((self.path.clone(), content))
    }
}
