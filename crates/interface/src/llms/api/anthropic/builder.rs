use super::{AnthropicBackend, AnthropicConfig};
use crate::llms::{
    LLMBackend,
    api::config::{ApiConfig, LLMApiConfigTrait},
};
use alith_devices::logging::{LoggingConfig, LoggingConfigTrait};
use alith_models::api_model::{ApiLLMModel, anthropic::AnthropicModelTrait};
use std::sync::Arc;

// Everything here can be implemented for any struct.
pub struct AnthropicBackendBuilder {
    pub config: AnthropicConfig,
    pub model: ApiLLMModel,
}

impl Default for AnthropicBackendBuilder {
    fn default() -> Self {
        Self {
            config: Default::default(),
            model: ApiLLMModel::claude_3_7_sonnet(),
        }
    }
}

impl AnthropicBackendBuilder {
    pub fn init(self) -> crate::Result<Arc<LLMBackend>> {
        Ok(Arc::new(LLMBackend::Anthropic(AnthropicBackend::new(
            self.config,
            self.model,
        )?)))
    }
}

impl LLMApiConfigTrait for AnthropicBackendBuilder {
    fn api_base_config_mut(&mut self) -> &mut ApiConfig {
        &mut self.config.api_config
    }

    fn api_config(&self) -> &ApiConfig {
        &self.config.api_config
    }
}

impl AnthropicModelTrait for AnthropicBackendBuilder {
    fn model(&mut self) -> &mut ApiLLMModel {
        &mut self.model
    }
}

impl LoggingConfigTrait for AnthropicBackendBuilder {
    fn logging_config_mut(&mut self) -> &mut LoggingConfig {
        &mut self.config.logging_config
    }
}
