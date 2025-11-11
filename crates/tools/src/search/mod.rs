use std::sync::Arc;

use alith_core::tool::{StructureTool, Tool, ToolError};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

pub mod duckduckgo;

pub use duckduckgo::SearchInput;

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
    type Input = SearchInput;
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
        self.searcher.author()
    }

    #[inline]
    async fn run_with_args(&self, input: Self::Input) -> Result<Self::Output, ToolError> {
        self.searcher
            .search(&input.query)
            .await
            .map_err(|err| ToolError::NormalError(Box::new(err)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_tool_schema_is_object() {
        let tool = SearchTool::default();
        let definition = StructureTool::definition(&tool);
        let schema = &definition.parameters;

        // Verify that parameters is an object type, not a string
        assert_eq!(
            schema.get("type").and_then(|v| v.as_str()),
            Some("object"),
            "SearchTool schema type should be 'object', not 'string'"
        );

        // Verify that the object has properties
        let properties = schema.get("properties");
        assert!(
            properties.is_some() && properties.unwrap().is_object(),
            "SearchTool schema should have properties"
        );

        // Verify that the query property exists
        let query_prop = schema
            .get("properties")
            .and_then(|props| props.get("query"));
        assert!(
            query_prop.is_some(),
            "SearchTool schema properties should have 'query' field"
        );
    }

    #[test]
    fn test_search_input_deserialize() {
        let json = r#"{"query": "Bitcoin"}"#;
        let input: Result<SearchInput, _> = serde_json::from_str(json);
        assert!(input.is_ok(), "SearchInput should deserialize from JSON");
        assert_eq!(input.unwrap().query, "Bitcoin");
    }
}
