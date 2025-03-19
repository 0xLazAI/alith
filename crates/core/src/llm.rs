pub mod client;

use crate::chat::{Completion, CompletionError};
use crate::embeddings::{Embeddings, EmbeddingsData, EmbeddingsError};
use anyhow::Result;
use async_trait::async_trait;
use client::{Client, CompletionResponse};

#[cfg(feature = "inference")]
use fastembed::TextEmbedding;
#[cfg(feature = "inference")]
pub use fastembed::{
    EmbeddingModel as FastEmbeddingsModelName, ExecutionProviderDispatch,
    InitOptions as FastEmbeddingsModelOptions,
};
#[cfg(feature = "inference")]
use std::sync::Arc;

// OpenAI models

pub const GPT_4_5: &str = "gpt-4.5";
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
pub const CLAUDE_3_7_SONNET: &str = "claude-3-7-sonnet";

// Remote Llama models

pub const LLAMA_3_1_SONAR_SMALL_ONLINE: &str = "llama-3.1-sonar-small-128k-online";
pub const LLAMA_3_1_SONAR_LARGE_ONLINE: &str = "llama-3.1-sonar-large-128k-online";
pub const LLAMA_3_1_SONAR_HUGE_ONLINE: &str = "llama-3.1-sonar-huge-128k-online";
pub const LLAMA_3_1_SONAR_SMALL_CHAT: &str = "llama-3.1-sonar-small-128k-chat";
pub const LLAMA_3_1_SONAR_LARGE_CHAT: &str = "llama-3.1-sonar-large-128k-chat";
pub const LLAMA_3_1_8B_INSTRUCT: &str = "llama-3.1-8b-instruct";
pub const LLAMA_3_1_70B_INSTRUCT: &str = "llama-3.1-70b-instruct";

// Remote Sonar models

pub const SONAR: &str = "sonar";
pub const SONAR_PRO: &str = "sonar-pro";
pub const SONAR_REASONING: &str = "sonar-reasoning";
pub const SONAR_REASONING_PRO: &str = "sonar-reasoning-pro";

pub mod openai_compatible {
    pub const DEEPSEEK_BASE_URL: &str = "https://api.deepseek.com";
    pub const GROQ_BASE_URL: &str = "https://api.groq.com/openai/v1";
    pub const HUNYUAN_BASE_URL: &str = "https://api.hunyuan.cloud.tencent.com/v1";
    pub const MINIMAX_BASE_URL: &str = "https://api.minimax.chat/v1";
    pub const MISTRAL_BASE_URL: &str = "https://api.mistral.ai/v1";
    pub const MOONSHOT_BASE_URL: &str = "https://api.moonshot.cn/v1";
    pub const PERPLEXITY_BASE_URL: &str = "https://api.perplexity.ai";
    pub const QIANWEN_BASE_URL: &str = "https://dashscope.aliyuncs.com/compatible-mode/v1";
}

/// A struct representing a Large Language Model (LLM)
pub struct LLM {
    /// The name or identifier of the model to use
    /// Examples: "gpt-4", "gpt-3.5-turbo", etc.
    pub model: String,
    /// The LLM client used to communicate with model backends
    client: Client,
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

    #[inline]
    pub fn client(&self) -> &Client {
        &self.client
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

#[derive(Clone)]
pub struct EmbeddingsModel {
    pub client: Client,
    pub model: String,
}

#[async_trait]
impl Embeddings for EmbeddingsModel {
    const MAX_DOCUMENTS: usize = 1024;

    async fn embed_texts(
        &self,
        input: Vec<String>,
    ) -> Result<Vec<EmbeddingsData>, EmbeddingsError> {
        self.client.embed_texts(&self.model, input).await
    }
}

#[cfg(feature = "inference")]
#[derive(Clone)]
pub struct FastEmbeddingsModel {
    model: Arc<TextEmbedding>,
}

#[cfg(feature = "inference")]
impl FastEmbeddingsModel {
    /// Try to generate a new TextEmbedding Instance.
    ///
    /// Uses the highest level of Graph optimization.
    ///
    /// Uses the total number of CPUs available as the number of intra-threads.
    pub fn try_new(opts: FastEmbeddingsModelOptions) -> anyhow::Result<Self> {
        let model = TextEmbedding::try_new(opts)?;
        Ok(Self {
            model: Arc::new(model),
        })
    }

    /// Try to generate a new TextEmbedding Instance.
    ///
    /// Uses the highest level of Graph optimization.
    ///
    /// Uses the total number of CPUs available as the number of intra-threads.
    #[inline]
    pub fn try_default() -> anyhow::Result<Self> {
        Self::try_new(Default::default())
    }
}

#[cfg(feature = "inference")]
#[async_trait]
impl Embeddings for FastEmbeddingsModel {
    const MAX_DOCUMENTS: usize = 1024;

    async fn embed_texts(
        &self,
        input: Vec<String>,
    ) -> Result<Vec<EmbeddingsData>, EmbeddingsError> {
        // Generate embeddings with the default batch size, 256
        let embeddings = self
            .model
            .embed(input.clone(), None)
            .map_err(|err| EmbeddingsError::ProviderError(err.to_string()))?;
        Ok(input
            .iter()
            .zip(embeddings)
            .map(|(doc, embeddings)| EmbeddingsData {
                document: doc.to_string(),
                vec: embeddings.iter().map(|e| *e as f64).collect(),
            })
            .collect())
    }
}
