use crate::requests::{
    completion::{
        error::CompletionError, request::CompletionRequest, response::CompletionResponse,
    },
    embeddings::{EmbeddingsError, EmbeddingsRequest, EmbeddingsResponse},
    logit_bias::LogitBias,
};
use alith_models::tokenizer::LlmTokenizer;
use alith_prompt::{LlmPrompt, PromptTokenizer};
pub mod api;
pub mod local;

pub enum LlmBackend {
    OpenAi(api::openai::OpenAiBackend),
    Anthropic(api::anthropic::AnthropicBackend),
    GenericApi(api::generic_openai::GenericApiBackend),
}

impl LlmBackend {
    pub(crate) async fn completion_request(
        &self,
        request: &CompletionRequest,
    ) -> crate::Result<CompletionResponse, CompletionError> {
        match self {
            LlmBackend::OpenAi(b) => b.completion_request(request).await,
            LlmBackend::Anthropic(b) => b.completion_request(request).await,
            LlmBackend::GenericApi(b) => b.completion_request(request).await,
        }
    }

    pub(crate) async fn embeddings_request(
        &self,
        request: &EmbeddingsRequest,
    ) -> crate::Result<EmbeddingsResponse, EmbeddingsError> {
        match self {
            LlmBackend::OpenAi(b) => b.embeddings_request(request).await,
            LlmBackend::GenericApi(b) => b.embeddings_request(request).await,
            _ => unimplemented!(),
        }
    }

    pub async fn clear_cache(
        self: &std::sync::Arc<Self>,
    ) -> crate::Result<CompletionResponse, CompletionError> {
        let mut request = CompletionRequest::new(std::sync::Arc::clone(self));
        request.config.cache_prompt = false;
        request.config.requested_response_tokens = Some(0);
        request.request().await
    }

    pub async fn set_cache(
        self: &std::sync::Arc<Self>,
        prompt: &LlmPrompt,
    ) -> crate::Result<CompletionResponse, CompletionError> {
        let mut request = CompletionRequest::new(std::sync::Arc::clone(self));
        request.config.cache_prompt = true;
        request.prompt = prompt.clone();
        request.config.requested_response_tokens = Some(0);
        request.request().await
    }

    pub fn new_prompt(&self) -> LlmPrompt {
        match self {
            LlmBackend::OpenAi(b) => LlmPrompt::new_openai_prompt(
                Some(b.model.tokens_per_message),
                b.model.tokens_per_name,
                self.prompt_tokenizer(),
            ),
            LlmBackend::Anthropic(b) => LlmPrompt::new_openai_prompt(
                Some(b.model.tokens_per_message),
                b.model.tokens_per_name,
                self.prompt_tokenizer(),
            ),
            LlmBackend::GenericApi(b) => LlmPrompt::new_openai_prompt(
                Some(b.model.tokens_per_message),
                b.model.tokens_per_name,
                self.prompt_tokenizer(),
            ),
        }
    }

    pub fn model_id(&self) -> &str {
        match self {
            LlmBackend::OpenAi(b) => &b.model.model_base.model_id,
            LlmBackend::Anthropic(b) => &b.model.model_base.model_id,
            LlmBackend::GenericApi(b) => &b.model.model_base.model_id,
        }
    }

    pub fn model_ctx_size(&self) -> u64 {
        match self {
            LlmBackend::OpenAi(b) => b.model.model_base.model_ctx_size,
            LlmBackend::Anthropic(b) => b.model.model_base.model_ctx_size,
            LlmBackend::GenericApi(b) => b.model.model_base.model_ctx_size,
        }
    }

    pub fn inference_ctx_size(&self) -> u64 {
        match self {
            LlmBackend::OpenAi(b) => b.model.model_base.inference_ctx_size,
            LlmBackend::Anthropic(b) => b.model.model_base.inference_ctx_size,
            LlmBackend::GenericApi(b) => b.model.model_base.inference_ctx_size,
        }
    }

    pub fn tokenizer(&self) -> &std::sync::Arc<LlmTokenizer> {
        match self {
            LlmBackend::OpenAi(b) => &b.model.model_base.tokenizer,
            LlmBackend::Anthropic(b) => &b.model.model_base.tokenizer,
            LlmBackend::GenericApi(b) => &b.model.model_base.tokenizer,
        }
    }

    fn prompt_tokenizer(&self) -> std::sync::Arc<dyn PromptTokenizer> {
        match self {
            LlmBackend::OpenAi(b) => std::sync::Arc::clone(&b.model.model_base.tokenizer)
                as std::sync::Arc<dyn PromptTokenizer>,
            LlmBackend::Anthropic(b) => std::sync::Arc::clone(&b.model.model_base.tokenizer)
                as std::sync::Arc<dyn PromptTokenizer>,
            LlmBackend::GenericApi(b) => std::sync::Arc::clone(&b.model.model_base.tokenizer)
                as std::sync::Arc<dyn PromptTokenizer>,
        }
    }

    pub fn build_logit_bias(&self, logit_bias: &mut Option<LogitBias>) -> crate::Result<()> {
        if let Some(logit_bias) = logit_bias {
            match self {
                LlmBackend::OpenAi(_) => logit_bias.build_openai(self.tokenizer())?,
                LlmBackend::Anthropic(_) => unreachable!("Anthropic does not support logit bias"),
                LlmBackend::GenericApi(_) => logit_bias.build_openai(self.tokenizer())?,
            };
        }
        Ok(())
    }

    pub fn openai(&self) -> crate::Result<&api::openai::OpenAiBackend> {
        match self {
            LlmBackend::OpenAi(b) => Ok(b),
            _ => crate::bail!("Backend is not openai"),
        }
    }

    pub fn anthropic(&self) -> crate::Result<&api::anthropic::AnthropicBackend> {
        match self {
            LlmBackend::Anthropic(b) => Ok(b),
            _ => crate::bail!("Backend is not anthropic"),
        }
    }

    pub fn generic_api(&self) -> crate::Result<&api::generic_openai::GenericApiBackend> {
        match self {
            LlmBackend::GenericApi(b) => Ok(b),
            _ => crate::bail!("Backend is not generic_api"),
        }
    }

    pub fn shutdown(&self) {
        match self {
            LlmBackend::OpenAi(_) => (),
            LlmBackend::Anthropic(_) => (),
            LlmBackend::GenericApi(_) => (),
        }
    }
}
