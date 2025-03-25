#[allow(unused_imports)]
pub(crate) use anyhow::{Error, Result, anyhow, bail};
#[allow(unused_imports)]
pub(crate) use tracing::{Level, debug, error, info, span, trace, warn};

pub mod llms;
pub mod requests;

use llms::api::anthropic::builder::AnthropicBackendBuilder;
use llms::api::openai::builder::OpenAIBackendBuilder;
use llms::api::perplexity::builder::PerplexityBackendBuilder;

pub use llms::{
    LLMBackend,
    api::client::ApiClient,
    api::config::{ApiConfig, ApiConfigTrait},
    local::{LLMLocalTrait, LocalLLMConfig},
};
pub use requests::{
    completion::{
        CompletionError, CompletionFinishReason, CompletionRequest, CompletionResponse,
        TimingUsage, TokenUsage, ToolChoice, ToolDefinition,
    },
    embeddings::{EmbeddingsData, EmbeddingsError, EmbeddingsRequest, EmbeddingsResponse},
    logit_bias::{LogitBias, LogitBiasTrait},
    req_components::{RequestConfig, RequestConfigTrait},
    res_components::{GenerationSettings, InferenceProbabilities, TopProbabilities},
    stop_sequence::{StopSequences, StoppingSequence},
};

pub struct LLMInterface;

// These are examples and bare minimum implementations. For full featured implementation see the alith-client crate.
impl LLMInterface {
    #[inline]
    pub fn openai() -> OpenAIBackendBuilder {
        OpenAIBackendBuilder::default()
    }

    #[inline]
    pub fn anthropic() -> AnthropicBackendBuilder {
        AnthropicBackendBuilder::default()
    }

    #[inline]
    pub fn perplexity() -> PerplexityBackendBuilder {
        PerplexityBackendBuilder::default()
    }
}
