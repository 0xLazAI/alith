use super::{GoogleBackend, GoogleConfig};
use crate::llms::{
    LLMBackend,
    api::config::{ApiConfig, LLMApiConfigTrait},
};
use alith_devices::logging::{LoggingConfig, LoggingConfigTrait};
use alith_models::api_model::{ApiLLMModel, google::GoogleModelTrait};
use std::sync::Arc;

pub struct GoogleBackendBuilder {
    pub config: GoogleConfig,
    pub model: ApiLLMModel,
}

impl Default for GoogleBackendBuilder {
    fn default() -> Self {
        Self {
            config: Default::default(),
            model: ApiLLMModel::gemini_2_5_flash(),
        }
    }
}

impl GoogleBackendBuilder {
    pub fn init(self) -> crate::Result<Arc<LLMBackend>> {
        Ok(Arc::new(LLMBackend::Google(GoogleBackend::new(
            self.config,
            self.model,
        )?)))
    }
}

impl LLMApiConfigTrait for GoogleBackendBuilder {
    fn api_base_config_mut(&mut self) -> &mut ApiConfig {
        &mut self.config.api_config
    }

    fn api_config(&self) -> &ApiConfig {
        &self.config.api_config
    }
}

impl GoogleModelTrait for GoogleBackendBuilder {
    fn model(&mut self) -> &mut ApiLLMModel {
        &mut self.model
    }
}

impl LoggingConfigTrait for GoogleBackendBuilder {
    fn logging_config_mut(&mut self) -> &mut LoggingConfig {
        &mut self.config.logging_config
    }
}
