use super::{Search, SearchError, SearchResult, SearchResults};
use alith_core::tool::{StructureTool, ToolError};
use async_trait::async_trait;
use reqwest::Client;
use scraper::{Html, Selector};
use std::collections::HashMap;
use url::Url;

pub const DEFAULT_URL: &str = "https://duckduckgo.com/html/";
pub const DEFAUTL_MAX_COUNT: usize = 10;

pub struct Searcher {
    url: String,
    client: Client,
    max_count: usize,
}

impl Default for Searcher {
    fn default() -> Self {
        Self {
            client: Client::new(),
            url: DEFAULT_URL.to_string(),
            max_count: DEFAUTL_MAX_COUNT,
        }
    }
}

#[async_trait]
impl Search for Searcher {
    async fn search(&self, query: &str) -> Result<SearchResults, SearchError> {
        let mut url = Url::parse(&self.url)?;

        let mut query_params = HashMap::new();
        query_params.insert("q", query);

        url.query_pairs_mut().extend_pairs(query_params.iter());

        let response = self.client.get(url).send().await?;
        let body = response.text().await?;
        let document = Html::parse_document(&body);

        let result_selector = Selector::parse(".web-result").unwrap();
        let result_title_selector = Selector::parse(".result__a").unwrap();
        let result_url_selector = Selector::parse(".result__url").unwrap();
        let result_snippet_selector = Selector::parse(".result__snippet").unwrap();

        let results = document
            .select(&result_selector)
            .map(|result| {
                let title = result
                    .select(&result_title_selector)
                    .next()
                    .unwrap()
                    .text()
                    .collect::<Vec<_>>()
                    .join("");
                let link = result
                    .select(&result_url_selector)
                    .next()
                    .unwrap()
                    .text()
                    .collect::<Vec<_>>()
                    .join("")
                    .trim()
                    .to_string();
                let snippet = result
                    .select(&result_snippet_selector)
                    .next()
                    .unwrap()
                    .text()
                    .collect::<Vec<_>>()
                    .join("");

                SearchResult {
                    title,
                    link,
                    snippet,
                }
            })
            .take(self.max_count)
            .collect::<Vec<_>>();

        Ok(results)
    }
}

#[async_trait]
impl StructureTool for Searcher {
    type Input = String;
    type Output = SearchResults;

    fn name(&self) -> &str {
        "DuckDuckGoSearch"
    }

    fn description(&self) -> &str {
        r#"DuckDuckGoSearch is a tool designed to perform search queries on the DuckDuckGo search engine.
It takes a search query string as input and returns relevant search results.
This tool is ideal for scenarios where real-time information from the internet is required,
such as finding the latest news, retrieving detailed information on a specific topic, or verifying facts.
"#
    }

    async fn run_with_args(&self, input: Self::Input) -> Result<Self::Output, ToolError> {
        self.search(&input)
            .await
            .map_err(|err| ToolError::NormalError(Box::new(err)))
    }
}
