use crate::requests::{
    completion::{
        error::CompletionError, request::CompletionRequest, response::CompletionResponse,
    },
    embeddings::{EmbeddingsError, EmbeddingsRequest, EmbeddingsResponse},
    logit_bias::LogitBias,
};
use alith_models::tokenizer::Tokenizer;
use alith_prompt::{LLMPrompt, PromptTokenizer};
pub mod api;
pub mod local;
use api::anthropic::AnthropicBackend;
use api::generic_openai::GenericApiBackend;
use api::openai::OpenAIBackend;
use std::sync::Arc;

pub enum LLMBackend {
    OpenAI(OpenAIBackend),
    Anthropic(AnthropicBackend),
    GenericApi(GenericApiBackend),
}

impl LLMBackend {
    pub(crate) async fn completion_request(
        &self,
        request: &CompletionRequest,
    ) -> crate::Result<CompletionResponse, CompletionError> {
        match self {
            LLMBackend::OpenAI(b) => b.completion_request(request).await,
            LLMBackend::Anthropic(b) => b.completion_request(request).await,
            LLMBackend::GenericApi(b) => b.completion_request(request).await,
        }
    }

    pub(crate) async fn embeddings_request(
        &self,
        request: &EmbeddingsRequest,
    ) -> crate::Result<EmbeddingsResponse, EmbeddingsError> {
        match self {
            LLMBackend::OpenAI(b) => b.embeddings_request(request).await,
            LLMBackend::GenericApi(b) => b.embeddings_request(request).await,
            _ => unimplemented!(),
        }
    }

    pub async fn clear_cache(
        self: &Arc<Self>,
    ) -> crate::Result<CompletionResponse, CompletionError> {
        let mut request = CompletionRequest::new(Arc::clone(self));
        request.config.cache_prompt = false;
        request.config.requested_response_tokens = Some(0);
        request.request().await
    }

    pub async fn set_cache(
        self: &Arc<Self>,
        prompt: &LLMPrompt,
    ) -> crate::Result<CompletionResponse, CompletionError> {
        let mut request = CompletionRequest::new(Arc::clone(self));
        request.config.cache_prompt = true;
        request.prompt = prompt.clone();
        request.config.requested_response_tokens = Some(0);
        request.request().await
    }

    pub fn new_prompt(&self) -> LLMPrompt {
        match self {
            LLMBackend::OpenAI(b) => LLMPrompt::new_api_prompt(
                self.prompt_tokenizer(),
                Some(b.model.tokens_per_message),
                b.model.tokens_per_name,
            ),
            LLMBackend::Anthropic(b) => LLMPrompt::new_api_prompt(
                self.prompt_tokenizer(),
                Some(b.model.tokens_per_message),
                b.model.tokens_per_name,
            ),
            LLMBackend::GenericApi(b) => LLMPrompt::new_api_prompt(
                self.prompt_tokenizer(),
                Some(b.model.tokens_per_message),
                b.model.tokens_per_name,
            ),
        }
    }

    pub fn get_total_prompt_tokens(&self, prompt: &LLMPrompt) -> crate::Result<usize> {
        match self {
            LLMBackend::OpenAI(_) => prompt.api_prompt()?.get_total_prompt_tokens(),
            LLMBackend::Anthropic(_) => prompt.api_prompt()?.get_total_prompt_tokens(),
            LLMBackend::GenericApi(_) => prompt.api_prompt()?.get_total_prompt_tokens(),
        }
    }

    pub fn model_id(&self) -> &str {
        match self {
            LLMBackend::OpenAI(b) => &b.model.model_base.model_id,
            LLMBackend::Anthropic(b) => &b.model.model_base.model_id,
            LLMBackend::GenericApi(b) => &b.model.model_base.model_id,
        }
    }

    pub fn model_ctx_size(&self) -> u64 {
        match self {
            LLMBackend::OpenAI(b) => b.model.model_base.model_ctx_size,
            LLMBackend::Anthropic(b) => b.model.model_base.model_ctx_size,
            LLMBackend::GenericApi(b) => b.model.model_base.model_ctx_size,
        }
    }

    pub fn inference_ctx_size(&self) -> u64 {
        match self {
            LLMBackend::OpenAI(b) => b.model.model_base.inference_ctx_size,
            LLMBackend::Anthropic(b) => b.model.model_base.inference_ctx_size,
            LLMBackend::GenericApi(b) => b.model.model_base.inference_ctx_size,
        }
    }

    pub fn tokenizer(&self) -> &Arc<Tokenizer> {
        match self {
            LLMBackend::OpenAI(b) => &b.model.model_base.tokenizer,
            LLMBackend::Anthropic(b) => &b.model.model_base.tokenizer,
            LLMBackend::GenericApi(b) => &b.model.model_base.tokenizer,
        }
    }

    fn prompt_tokenizer(&self) -> Arc<dyn PromptTokenizer> {
        match self {
            LLMBackend::OpenAI(b) => {
                Arc::clone(&b.model.model_base.tokenizer) as Arc<dyn PromptTokenizer>
            }
            LLMBackend::Anthropic(b) => {
                Arc::clone(&b.model.model_base.tokenizer) as Arc<dyn PromptTokenizer>
            }
            LLMBackend::GenericApi(b) => {
                Arc::clone(&b.model.model_base.tokenizer) as Arc<dyn PromptTokenizer>
            }
        }
    }

    pub fn build_logit_bias(&self, logit_bias: &mut Option<LogitBias>) -> crate::Result<()> {
        if let Some(logit_bias) = logit_bias {
            match self {
                LLMBackend::OpenAI(_) => logit_bias.build_openai(self.tokenizer())?,
                LLMBackend::Anthropic(_) => unreachable!("Anthropic does not support logit bias"),
                LLMBackend::GenericApi(_) => logit_bias.build_openai(self.tokenizer())?,
            };
        }
        Ok(())
    }

    pub fn openai(&self) -> crate::Result<&api::openai::OpenAIBackend> {
        match self {
            LLMBackend::OpenAI(b) => Ok(b),
            _ => crate::bail!("Backend is not openai"),
        }
    }

    pub fn anthropic(&self) -> crate::Result<&api::anthropic::AnthropicBackend> {
        match self {
            LLMBackend::Anthropic(b) => Ok(b),
            _ => crate::bail!("Backend is not anthropic"),
        }
    }

    pub fn generic_api(&self) -> crate::Result<&api::generic_openai::GenericApiBackend> {
        match self {
            LLMBackend::GenericApi(b) => Ok(b),
            _ => crate::bail!("Backend is not generic_api"),
        }
    }

    pub fn shutdown(&self) {
        match self {
            LLMBackend::OpenAI(_) => (),
            LLMBackend::Anthropic(_) => (),
            LLMBackend::GenericApi(_) => (),
        }
    }
}
