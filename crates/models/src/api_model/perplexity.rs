use super::ApiLLMModel;
use crate::{LLMModelBase, tokenizer::Tokenizer};
use std::sync::Arc;

impl ApiLLMModel {
    pub fn perplexity_model_from_model_id(model_id: &str) -> ApiLLMModel {
        if model_id.starts_with("llama-3.1-sonar-small") {
            Self::sonar_small()
        } else if model_id.starts_with("llama-3.1-sonar-large") {
            Self::sonar_large()
        } else if model_id.starts_with("llama-3.1-sonar-huge") {
            Self::sonar_huge()
        } else if model_id.contains("sonar-small") {
            Self::sonar_small()
        } else if model_id.contains("sonar-large") {
            Self::sonar_large()
        } else if model_id.contains("sonar-huge") {
            Self::sonar_huge()
        } else {
            panic!("Model ID ({model_id}) not found for ApiLLMModel")
        }
    }

    pub fn sonar_small() -> ApiLLMModel {
        let model_id = "llama-3.1-sonar-small-128k-online".to_string();
        let tokenizer = model_tokenizer(&model_id);
        ApiLLMModel {
            model_base: LLMModelBase {
                model_id,
                model_ctx_size: 127072,
                inference_ctx_size: 8192,
                tokenizer,
            },
            cost_per_m_in_tokens: 0.1,
            cost_per_m_out_tokens: 0.1,
            tokens_per_message: 3,
            tokens_per_name: None,
        }
    }

    pub fn sonar_large() -> ApiLLMModel {
        let model_id = "llama-3.1-sonar-large-128k-online".to_string();
        let tokenizer = model_tokenizer(&model_id);
        ApiLLMModel {
            model_base: LLMModelBase {
                model_id,
                model_ctx_size: 127072,
                inference_ctx_size: 8192,
                tokenizer,
            },
            cost_per_m_in_tokens: 0.5,
            cost_per_m_out_tokens: 0.5,
            tokens_per_message: 3,
            tokens_per_name: None,
        }
    }

    pub fn sonar_huge() -> ApiLLMModel {
        let model_id = "llama-3.1-sonar-huge-128k-online".to_string();
        let tokenizer = model_tokenizer(&model_id);
        ApiLLMModel {
            model_base: LLMModelBase {
                model_id,
                model_ctx_size: 127072,
                inference_ctx_size: 8192,
                tokenizer,
            },
            cost_per_m_in_tokens: 2.5,
            cost_per_m_out_tokens: 2.5,
            tokens_per_message: 3,
            tokens_per_name: None,
        }
    }
}

pub fn model_tokenizer(_model_id: &str) -> Arc<Tokenizer> {
    Arc::new(
        Tokenizer::new_tiktoken("gpt-4")
            .unwrap_or_else(|_| panic!("Failed to load tokenizer for gpt-4")),
    )
}

pub trait PerplexityModelTrait: Sized {
    fn model(&mut self) -> &mut ApiLLMModel;

    /// Set the model using the model_id string.
    fn model_id_str(mut self, model_id: &str) -> Self
    where
        Self: Sized,
    {
        *self.model() = ApiLLMModel::perplexity_model_from_model_id(model_id);
        self
    }

    fn sonar_small(mut self) -> Self
    where
        Self: Sized,
    {
        *self.model() = ApiLLMModel::sonar_small();
        self
    }

    fn sonar_large(mut self) -> Self
    where
        Self: Sized,
    {
        *self.model() = ApiLLMModel::sonar_large();
        self
    }

    fn sonar_huge(mut self) -> Self
    where
        Self: Sized,
    {
        *self.model() = ApiLLMModel::sonar_huge();
        self
    }
}
