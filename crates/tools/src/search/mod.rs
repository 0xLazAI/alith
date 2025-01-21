use std::sync::Arc;

use alith_core::tool::{StructureTool, Tool, ToolError};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

pub mod duckduckgo;

#[derive(Debug, Default)]
pub enum SearchProvider {
    #[default]
    DuckDuckGo,
}

#[derive(Debug, thiserror::Error)]
#[error("Search error")]
pub enum SearchError {
    #[error("Failed to search: {0}")]
    SearchError(String),
    #[error("An unknown error occurred: {0}")]
    Unknown(String),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Url error: {0}")]
    UrlError(#[from] url::ParseError),
    #[error("Request error: {0}")]
    RequestError(#[from] reqwest::Error),
    /// JSON error (e.g.: serialization, deserialization, etc.)
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    title: String,
    link: String,
    snippet: String,
}

pub type SearchResults = Vec<SearchResult>;

#[async_trait]
pub trait Search: Tool {
    async fn search(&self, query: &str) -> Result<SearchResults, SearchError>;
}

pub struct SearchTool {
    provider: SearchProvider,
    searcher: Arc<dyn Search>,
}

impl Default for SearchTool {
    fn default() -> Self {
        let provider = SearchProvider::default();
        Self {
            searcher: Self::searcher(&provider),
            provider,
        }
    }
}

impl SearchTool {
    #[inline]
    pub fn provider(&self) -> &SearchProvider {
        &self.provider
    }

    #[inline]
    pub fn searcher(provider: &SearchProvider) -> Arc<dyn Search> {
        Arc::new(match provider {
            SearchProvider::DuckDuckGo => duckduckgo::Searcher::default(),
        })
    }
}

#[async_trait]
impl StructureTool for SearchTool {
    type Input = String;
    type Output = SearchResults;

    #[inline]
    fn name(&self) -> &str {
        self.searcher.name()
    }

    #[inline]
    fn description(&self) -> &str {
        self.searcher.description()
    }

    #[inline]
    fn version(&self) -> &str {
        self.searcher.version()
    }

    #[inline]
    fn author(&self) -> &str {
        self.searcher.version()
    }

    #[inline]
    async fn run_with_args(&self, input: Self::Input) -> Result<Self::Output, ToolError> {
        self.searcher
            .search(&input)
            .await
            .map_err(|err| ToolError::NormalError(Box::new(err)))
    }
}
