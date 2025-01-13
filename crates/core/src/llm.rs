pub mod client;

use crate::chat::{Completion, CompletionError};
use crate::embeddings::{Embeddings, EmbeddingsData, EmbeddingsError};
use anyhow::Result;
use async_trait::async_trait;
use client::{Client, CompletionResponse};

// OpenAI models

pub const GPT_4: &str = "gpt-4";
pub const GPT_4_32K: &str = "gpt-4-32k";
pub const GPT_4_TURBO: &str = "gpt-4-turbo";
pub const GPT_3_5_TURBO: &str = "gpt-3.5-turbo";
pub const GPT_4O_MINI: &str = "gpt-4o-mini";

// Anthropic models

pub const CLAUDE_3_OPUS: &str = "claude-3-opus";
pub const CLAUDE_3_SONNET: &str = "claude-3-sonnet";
pub const CLAUDE_3_HAIKU: &str = "claude-3-haiku";
pub const CLAUDE_3_5_SONNET: &str = "claude-3-5-sonnet";

// Remote Llama models

pub const LLAMA_3_1_SONAR_SMALL_ONLINE: &str = "llama-3.1-sonar-small-128k-online";
pub const LLAMA_3_1_SONAR_LARGE_ONLINE: &str = "llama-3.1-sonar-large-128k-online";
pub const LLAMA_3_1_SONAR_HUGE_ONLINE: &str = "llama-3.1-sonar-huge-128k-online";
pub const LLAMA_3_1_SONAR_SMALL_CHAT: &str = "llama-3.1-sonar-small-128k-chat";
pub const LLAMA_3_1_SONAR_LARGE_CHAT: &str = "llama-3.1-sonar-large-128k-chat";
pub const LLAMA_3_1_8B_INSTRUCT: &str = "llama-3.1-8b-instruct";
pub const LLAMA_3_1_70B_INSTRUCT: &str = "llama-3.1-70b-instruct";

/// A struct representing a Large Language Model (LLM)
pub struct LLM {
    /// The name or identifier of the model to use
    /// Examples: "gpt-4", "gpt-3.5-turbo", etc.
    pub model: String,
    /// The LLM client used to communicate with model backends
    client: Client,
}

#[derive(Clone)]
pub struct EmbeddingsModel {
    pub client: Client,
    pub model: String,
}

impl LLM {
    pub fn from_model_name(model: &str) -> Result<Self> {
        Ok(Self {
            model: model.to_string(),
            client: Client::from_model_name(model)?,
        })
    }

    pub fn openai_compatible_model(api_key: &str, base_url: &str, model: &str) -> Result<Self> {
        Ok(Self {
            model: model.to_string(),
            client: Client::openai_compatible_client(api_key, base_url, model)?,
        })
    }

    pub fn embeddings_model(&self, model: &str) -> EmbeddingsModel {
        EmbeddingsModel {
            model: model.to_string(),
            client: self.client.clone(),
        }
    }
}

impl Completion for LLM {
    type Response = CompletionResponse;

    async fn completion(
        &mut self,
        request: crate::chat::Request,
    ) -> Result<Self::Response, CompletionError> {
        self.client.completion(request).await
    }
}

#[async_trait]
impl Embeddings for EmbeddingsModel {
    const MAX_DOCUMENTS: usize = 1024;

    async fn embed_texts(
        &self,
        input: Vec<String>,
    ) -> Result<Vec<EmbeddingsData>, EmbeddingsError> {
        self.client.embed_texts(input).await
    }
}
