use html_to_markdown::{TagHandler, markdown};
use std::io::Read;
use std::sync::RwLock;
use std::{cell::RefCell, rc::Rc};
use url::Url;

use alith_core::{
    chunking::{ChunkError, Chunker, chunk_text},
    knowledge::{Knowledge, KnowledgeError},
};

pub struct HtmlKnowledge<R> {
    html: RwLock<R>,
    url: Url,
    to_markdown: bool,
}

impl<R: Read> HtmlKnowledge<R> {
    pub fn new(html: R, url: Url, to_markdown: bool) -> Self {
        Self {
            html: RwLock::new(html),
            url,
            to_markdown,
        }
    }
}

impl<R: Read + Send + Sync> Chunker for HtmlKnowledge<R> {
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

impl<R: Read + Sync + Send> Knowledge for HtmlKnowledge<R> {
    fn load(&self) -> Result<String, KnowledgeError> {
        let mut html = self
            .html
            .write()
            .map_err(|err| KnowledgeError::LoadError(err.to_string()))?;
        let mut html = html.by_ref();
        let cleaned_html = readability::extractor::extract(&mut html, &self.url)
            .map_err(|err| KnowledgeError::LoadError(err.to_string()))?;
        let html = format!("{}\n{}", cleaned_html.title, cleaned_html.text);
        if self.to_markdown {
            Ok(html_to_md(&html))
        } else {
            Ok(html)
        }
    }

    fn enrich(&self, _input: &str) -> Result<String, KnowledgeError> {
        Ok(format!("<html>{}</html>", self.load()?))
    }
}

/// Converts the provided HTML string to Markdown string.
pub fn html_to_md(html: &str) -> String {
    let mut handlers: Vec<TagHandler> = vec![
        Rc::new(RefCell::new(markdown::ParagraphHandler)),
        Rc::new(RefCell::new(markdown::HeadingHandler)),
        Rc::new(RefCell::new(markdown::ListHandler)),
        Rc::new(RefCell::new(markdown::TableHandler::new())),
        Rc::new(RefCell::new(markdown::StyledTextHandler)),
        Rc::new(RefCell::new(markdown::CodeHandler)),
        Rc::new(RefCell::new(markdown::WebpageChromeRemover)),
    ];

    html_to_markdown::convert_html_to_markdown(html.as_bytes(), &mut handlers)
        .unwrap_or_else(|_| html.to_string())
}
