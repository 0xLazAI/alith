use crate::LLMClient;
use alith_devices::logging::{LoggingConfig, LoggingConfigTrait};
use alith_interface::llms::{
    LLMBackend,
    api::{
        config::{ApiConfig, LLMApiConfigTrait},
        openai::{OpenAIBackend, OpenAIConfig},
    },
};
use alith_models::api_model::{ApiLLMModel, openai::OpenAIModelTrait};
use std::sync::Arc;

// Everything here can be implemented for any struct.
pub struct OpenAIBackendBuilder {
    pub config: OpenAIConfig,
    pub model: ApiLLMModel,
}

impl Default for OpenAIBackendBuilder {
    fn default() -> Self {
        Self {
            config: Default::default(),
            model: ApiLLMModel::gpt_4_o_mini(),
        }
    }
}

impl OpenAIBackendBuilder {
    pub fn init(self) -> crate::Result<LLMClient> {
        Ok(LLMClient::new(Arc::new(LLMBackend::OpenAI(
            OpenAIBackend::new(self.config, self.model)?,
        ))))
    }
}

impl LLMApiConfigTrait for OpenAIBackendBuilder {
    fn api_base_config_mut(&mut self) -> &mut ApiConfig {
        &mut self.config.api_config
    }

    fn api_config(&self) -> &ApiConfig {
        &self.config.api_config
    }
}

impl OpenAIModelTrait for OpenAIBackendBuilder {
    fn model(&mut self) -> &mut ApiLLMModel {
        &mut self.model
    }
}

impl LoggingConfigTrait for OpenAIBackendBuilder {
    fn logging_config_mut(&mut self) -> &mut LoggingConfig {
        &mut self.config.logging_config
    }
}
