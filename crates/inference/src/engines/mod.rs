#[cfg(feature = "ort")]
pub mod ort;
// #[cfg(feature = "llamacpp")]
pub mod llamacpp;
#[cfg(feature = "python")]
pub mod python;
// #[cfg(feature = "mistralrs")]
pub mod mistralrs;

use alith_models::{TokenIdType, TokenType};
use anyhow::{Result, anyhow as error};
use serde::{Deserialize, Serialize};

/// [`Request`] is the internal representation of an LLM request. The [`dynamo.llm-preprocessor`]
/// crate is responsible for converting request from the public APIs to this internal representation.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Request {
    /// Type of prompt
    pub token_ids: Vec<TokenIdType>,
    pub max_tokens: Option<u32>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct EngineOutput {
    /// new token_ids
    pub token_ids: Vec<TokenIdType>,
    /// If the LLM Engine performs the detokenization, then this will have a Some of the detokenized
    /// text/tokens. If this value is None, then the Backend is responsible for detokenization.
    pub tokens: Option<Vec<TokenType>>,
    /// Decoded text
    pub text: Option<String>,
    /// Finish reason
    pub finish_reason: Option<FinishReason>,
}

impl EngineOutput {
    pub fn cancelled() -> Self {
        EngineOutput {
            token_ids: vec![],
            tokens: None,
            text: None,
            finish_reason: Some(FinishReason::Cancelled),
        }
    }

    pub fn stop() -> Self {
        EngineOutput {
            token_ids: vec![],
            tokens: None,
            text: None,
            finish_reason: Some(FinishReason::Stop),
        }
    }

    pub fn length() -> Self {
        EngineOutput {
            token_ids: vec![],
            tokens: None,
            text: None,
            finish_reason: Some(FinishReason::Length),
        }
    }

    pub fn error(err_msg: String) -> Self {
        EngineOutput {
            token_ids: vec![],
            tokens: None,
            text: None,
            finish_reason: Some(FinishReason::Error(err_msg)),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub enum FinishReason {
    #[serde(rename = "eos")]
    EoS,
    #[serde(rename = "length")]
    Length,
    #[serde(rename = "stop")]
    Stop,
    #[serde(rename = "error")]
    Error(String),
    #[serde(rename = "cancelled")]
    Cancelled,
}

impl std::fmt::Display for FinishReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FinishReason::EoS => write!(f, "eos"),
            FinishReason::Length => write!(f, "length"),
            FinishReason::Stop => write!(f, "stop"),
            FinishReason::Error(msg) => write!(f, "error: {}", msg),
            FinishReason::Cancelled => write!(f, "cancelled"),
        }
    }
}

/// Our services have the option of returning an "annotated" stream, which allows use
/// to include additional information with each delta. This is useful for debugging,
/// performance benchmarking, and improved observability.
#[derive(Serialize, Deserialize, Debug)]
pub struct Annotated<R> {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<R>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub event: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub comment: Option<Vec<String>>,
}

impl<R> Annotated<R> {
    /// Create a new annotated stream from the given error
    pub fn from_error(error: String) -> Self {
        Self {
            data: None,
            id: None,
            event: Some("error".to_string()),
            comment: Some(vec![error]),
        }
    }

    /// Create a new annotated stream from the given data
    pub fn from_data(data: R) -> Self {
        Self {
            data: Some(data),
            id: None,
            event: None,
            comment: None,
        }
    }

    /// Add an annotation to the stream
    ///
    /// Annotations populate the `event` field and the `comment` field
    pub fn from_annotation<S: Serialize>(
        name: impl Into<String>,
        value: &S,
    ) -> Result<Self, serde_json::Error> {
        Ok(Self {
            data: None,
            id: None,
            event: Some(name.into()),
            comment: Some(vec![serde_json::to_string(value)?]),
        })
    }

    /// Convert to a [`Result<Self, String>`]
    /// If [`Self::event`] is "error", return an error message(s) held by [`Self::comment`]
    pub fn ok(self) -> Result<Self, String> {
        if let Some(event) = &self.event {
            if event == "error" {
                return Err(self
                    .comment
                    .unwrap_or(vec!["unknown error".to_string()])
                    .join(", "));
            }
        }
        Ok(self)
    }

    pub fn is_ok(&self) -> bool {
        self.event.as_deref() != Some("error")
    }

    pub fn is_err(&self) -> bool {
        !self.is_ok()
    }

    pub fn is_event(&self) -> bool {
        self.event.is_some()
    }

    pub fn transfer<U: Serialize>(self, data: Option<U>) -> Annotated<U> {
        Annotated::<U> {
            data,
            id: self.id,
            event: self.event,
            comment: self.comment,
        }
    }

    /// Apply a mapping/transformation to the data field
    /// If the mapping fails, the error is returned as an annotated stream
    pub fn map_data<U, F>(self, transform: F) -> Annotated<U>
    where
        F: FnOnce(R) -> Result<U, String>,
    {
        match self.data.map(transform).transpose() {
            Ok(data) => Annotated::<U> {
                data,
                id: self.id,
                event: self.event,
                comment: self.comment,
            },
            Err(e) => Annotated::from_error(e),
        }
    }

    pub fn is_error(&self) -> bool {
        self.event.as_deref() == Some("error")
    }

    pub fn into_result(self) -> Result<Option<R>> {
        match self.data {
            Some(data) => Ok(Some(data)),
            None => match self.event {
                Some(event) if event == "error" => Err(error!(
                    self.comment
                        .unwrap_or(vec!["unknown error".to_string()])
                        .join(", ")
                ))?,
                _ => Ok(None),
            },
        }
    }
}
