use crate::LLMClient;
use alith_devices::logging::{LoggingConfig, LoggingConfigTrait};
use alith_interface::llms::{
    LLMBackend,
    api::{
        config::{ApiConfig, LLMApiConfigTrait},
        generic_openai::{GenericApiBackend, GenericApiConfig},
    },
};
use alith_models::api_model::{ApiLLMModel, perplexity::PerplexityModelTrait};
use std::sync::Arc;

// Everything here can be implemented for any struct.
pub struct PerplexityBackendBuilder {
    pub config: GenericApiConfig,
    pub model: ApiLLMModel,
}

impl Default for PerplexityBackendBuilder {
    fn default() -> Self {
        let mut config = GenericApiConfig::default();
        config.api_config.host = "api.perplexity.ai".to_string();
        config.api_config.api_key_env_var = "PERPLEXITY_API_KEY".to_string();
        config.logging_config.logger_name = "perplexity".to_string();
        Self {
            config,
            model: ApiLLMModel::sonar_large(),
        }
    }
}

impl PerplexityBackendBuilder {
    pub fn init(self) -> crate::Result<LLMClient> {
        Ok(LLMClient::new(Arc::new(LLMBackend::GenericApi(
            GenericApiBackend::new(self.config, self.model)?,
        ))))
    }
}

impl PerplexityModelTrait for PerplexityBackendBuilder {
    fn model(&mut self) -> &mut ApiLLMModel {
        &mut self.model
    }
}

impl LLMApiConfigTrait for PerplexityBackendBuilder {
    fn api_base_config_mut(&mut self) -> &mut ApiConfig {
        &mut self.config.api_config
    }

    fn api_config(&self) -> &ApiConfig {
        &self.config.api_config
    }
}

impl LoggingConfigTrait for PerplexityBackendBuilder {
    fn logging_config_mut(&mut self) -> &mut LoggingConfig {
        &mut self.config.logging_config
    }
}
