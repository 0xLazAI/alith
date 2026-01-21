use super::ApiLLMModel;
use crate::{LLMModelBase, tokenizer::Tokenizer};
use std::sync::Arc;

impl ApiLLMModel {
    pub fn google_model_from_model_id(model_id: &str) -> ApiLLMModel {
        if model_id.eq("gemini-2.5-pro") {
            Self::gemini_2_5_pro()
        } else if model_id.eq("gemini-2.5-flash") {
            Self::gemini_2_5_flash()
        } else if model_id.starts_with("gemini-2.5-flash-lite") {
            Self::gemini_2_5_flash_lite()
        } else if model_id.eq("gemini-2.0-flash") {
            Self::gemini_2_0_flash()
        } else if model_id.eq("gemini-2.0-flash-lite") {
            Self::gemini_2_0_flash_lite()
        } else {
            Self::gemini(model_id)
        }
    }

    pub fn gemini_2_5_pro() -> ApiLLMModel {
        let model_id = "gemini-2.5-pro".to_string();
        let tokenizer = model_tokenizer(&model_id);
        ApiLLMModel {
            model_base: LLMModelBase {
                model_id,
                model_ctx_size: 1_048_576,
                inference_ctx_size: 65_536,
                tokenizer,
            },
            cost_per_m_in_tokens: 1.25,   // base tier
            cost_per_m_out_tokens: 10.00, // base tier
            tokens_per_message: 3,
            tokens_per_name: None,
        }
    }

    pub fn gemini_2_5_flash() -> ApiLLMModel {
        let model_id = "gemini-2.5-flash".to_string();
        let tokenizer = model_tokenizer(&model_id);
        ApiLLMModel {
            model_base: LLMModelBase {
                model_id,
                model_ctx_size: 1_048_576,
                inference_ctx_size: 65_536,
                tokenizer,
            },
            cost_per_m_in_tokens: 0.30,
            cost_per_m_out_tokens: 2.50,
            tokens_per_message: 3,
            tokens_per_name: None,
        }
    }

    pub fn gemini_2_5_flash_lite() -> ApiLLMModel {
        let model_id = "gemini-2.5-flash-lite-preview-06-17".to_string();
        let tokenizer = model_tokenizer(&model_id);
        ApiLLMModel {
            model_base: LLMModelBase {
                model_id,
                model_ctx_size: 1_000_000,
                inference_ctx_size: 64_000,
                tokenizer,
            },
            cost_per_m_in_tokens: 0.10,
            cost_per_m_out_tokens: 0.40,
            tokens_per_message: 3,
            tokens_per_name: None,
        }
    }

    pub fn gemini_2_0_flash() -> ApiLLMModel {
        let model_id = "gemini-2.0-flash".to_string();
        let tokenizer = model_tokenizer(&model_id);
        ApiLLMModel {
            model_base: LLMModelBase {
                model_id,
                model_ctx_size: 1_048_576,
                inference_ctx_size: 8_192,
                tokenizer,
            },
            cost_per_m_in_tokens: 0.10,
            cost_per_m_out_tokens: 0.40,
            tokens_per_message: 3,
            tokens_per_name: None,
        }
    }

    pub fn gemini_2_0_flash_lite() -> ApiLLMModel {
        let model_id = "gemini-2.0-flash-lite".to_string();
        let tokenizer = model_tokenizer(&model_id);
        ApiLLMModel {
            model_base: LLMModelBase {
                model_id,
                model_ctx_size: 1_048_576,
                inference_ctx_size: 8_192,
                tokenizer,
            },
            cost_per_m_in_tokens: 0.075,
            cost_per_m_out_tokens: 0.30,
            tokens_per_message: 3,
            tokens_per_name: None,
        }
    }

    pub fn gemini<S: ToString>(model_id: S) -> ApiLLMModel {
        let model_id = model_id.to_string();
        let tokenizer = model_tokenizer(&model_id);
        ApiLLMModel {
            model_base: LLMModelBase {
                model_id,
                model_ctx_size: 1_048_576,
                inference_ctx_size: 65_536,
                tokenizer,
            },
            cost_per_m_in_tokens: 0.30,
            cost_per_m_out_tokens: 2.50,
            tokens_per_message: 3,
            tokens_per_name: None,
        }
    }
}

#[inline]
pub fn model_tokenizer(_model_id: &str) -> Arc<Tokenizer> {
    Arc::new(
        Tokenizer::new_tiktoken("gpt-4")
            .unwrap_or_else(|_| panic!("Failed to load tokenizer for gpt-4")),
    )
}

pub trait GoogleModelTrait: Sized {
    fn model(&mut self) -> &mut ApiLLMModel;

    /// Set the model using the model_id string.
    fn model_id_str(mut self, model_id: &str) -> Self
    where
        Self: Sized,
    {
        *self.model() = ApiLLMModel::google_model_from_model_id(model_id);
        self
    }

    /// Use the Gemini 2.5 pro model for the Google client.
    fn gemini_2_5_pro(mut self) -> Self
    where
        Self: Sized,
    {
        *self.model() = ApiLLMModel::gemini_2_5_pro();
        self
    }

    /// Use the Gemini 2.5 flash model for the Google client.
    fn gemini_2_5_flash(mut self) -> Self
    where
        Self: Sized,
    {
        *self.model() = ApiLLMModel::gemini_2_5_flash();
        self
    }

    /// Use the Gemini 2.5 flash lite model for the Google client.
    fn gemini_2_5_flash_lite(mut self) -> Self
    where
        Self: Sized,
    {
        *self.model() = ApiLLMModel::gemini_2_5_flash_lite();
        self
    }

    /// Use the Gemini 2.0 flash model for the Google client.
    fn gemini_2_0_flash(mut self) -> Self
    where
        Self: Sized,
    {
        *self.model() = ApiLLMModel::gemini_2_0_flash();
        self
    }

    /// Use the Gemini 2.0 flash lite model for the Google client.
    fn gemini_2_0_flash_lite(mut self) -> Self
    where
        Self: Sized,
    {
        *self.model() = ApiLLMModel::gemini_2_0_flash_lite();
        self
    }

    /// Use a custom Gemini model for the Google client.
    fn gemini<S: ToString>(mut self, model_id: S) -> Self
    where
        Self: Sized,
    {
        *self.model() = ApiLLMModel::gemini(model_id);
        self
    }
}
