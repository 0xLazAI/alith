use crate::json::parse_json_markdown;
use async_trait::async_trait;
use regex::{Error as RegexError, Regex};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ParserError {
    #[error("Regex error: {0}")]
    RegexError(#[from] RegexError),
    #[error("Parsing error: {0}")]
    ParsingError(String),
}

#[async_trait]
pub trait Parser: Send + Sync {
    async fn parse(&self, input: &str) -> Result<String, ParserError>;
}

#[derive(Debug, Clone, Default)]
pub struct StringParser;

#[async_trait]
impl Parser for StringParser {
    async fn parse(&self, input: &str) -> Result<String, ParserError> {
        Ok(input.to_string())
    }
}

#[derive(Debug, Clone)]
pub struct TrimParser {
    trim: bool,
}

impl TrimParser {
    pub fn new(trim: bool) -> Self {
        Self { trim }
    }
}

impl Default for TrimParser {
    fn default() -> Self {
        Self { trim: true }
    }
}

#[async_trait]
impl Parser for TrimParser {
    async fn parse(&self, input: &str) -> Result<String, ParserError> {
        if self.trim {
            Ok(input.trim().to_string())
        } else {
            Ok(input.to_string())
        }
    }
}

#[derive(Debug, Clone)]
pub struct MarkdownParser {
    regex: String,
    trim: bool,
}

impl MarkdownParser {
    pub fn new() -> Self {
        Self {
            regex: r"```(?:\w+)?\s*([\s\S]+?)\s*```".to_string(),
            trim: false,
        }
    }

    pub fn custom_expresion(mut self, regex: &str) -> Self {
        self.regex = regex.to_string();
        self
    }

    pub fn trim(mut self, trim: bool) -> Self {
        self.trim = trim;
        self
    }
}

impl Default for MarkdownParser {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Parser for MarkdownParser {
    async fn parse(&self, input: &str) -> Result<String, ParserError> {
        let re = Regex::new(&self.regex)?;
        if let Some(cap) = re.captures(input) {
            let find = cap[1].to_string();
            if self.trim {
                Ok(find.trim().to_string())
            } else {
                Ok(find)
            }
        } else {
            Err(ParserError::ParsingError(
                "No markdown code block found".into(),
            ))
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct JsonParser {
    pretty: bool,
}

impl JsonParser {
    pub fn new() -> Self {
        Self { pretty: false }
    }

    pub fn pretty(mut self, pretty: bool) -> Self {
        self.pretty = pretty;
        self
    }
}

#[async_trait]
impl Parser for JsonParser {
    async fn parse(&self, input: &str) -> Result<String, ParserError> {
        match parse_json_markdown(input)
            .map(|v| {
                if self.pretty {
                    serde_json::to_string_pretty(&v)
                } else {
                    serde_json::to_string(&v)
                }
            })
            .map_err(|err| ParserError::ParsingError(err.to_string()))
        {
            Ok(r) => r.map_err(|err| ParserError::ParsingError(err.to_string())),
            Err(err) => Err(err),
        }
    }
}
