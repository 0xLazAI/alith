pub mod builder;

use super::{
    client::ApiClient,
    config::{ApiConfig, ApiConfigTrait},
    openai::completion::OpenAICompletionRequest, // Format same as OpenAI
};
use crate::requests::{
    completion::{
        error::CompletionError, request::CompletionRequest, response::CompletionResponse,
    },
    embeddings::{EmbeddingsError, EmbeddingsRequest, EmbeddingsResponse},
};
use alith_devices::logging::LoggingConfig;
use alith_models::api_model::ApiLLMModel;
// use completion::GoogleCompletionRequest;
use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderValue};
use secrecy::{ExposeSecret, SecretString};
use serde_json::json;

/// Default gemini openai compatible v1beta api base url
pub const GOOGLE_API_BASE: &str = "https://generativelanguage.googleapis.com/v1beta/openai";

pub struct GoogleBackend {
    pub(crate) client: ApiClient<GoogleConfig>,
    pub model: ApiLLMModel,
}

impl GoogleBackend {
    pub fn new(mut config: GoogleConfig, model: ApiLLMModel) -> crate::Result<Self> {
        config.logging_config.load_logger()?;
        config.api_config.api_key = Some(config.api_config.load_api_key()?);
        Ok(Self {
            client: ApiClient::new(config),
            model,
        })
    }

    pub(crate) async fn completion_request(
        &self,
        request: &CompletionRequest,
    ) -> crate::Result<CompletionResponse, CompletionError> {
        match self
            .client
            .post("/chat/completions", OpenAICompletionRequest::new(request)?)
            .await
        {
            Err(e) => Err(CompletionError::ClientError(e)),
            Ok(res) => Ok(CompletionResponse::new_from_openai(request, res)?),
        }
    }

    pub(crate) async fn embeddings_request(
        &self,
        request: &EmbeddingsRequest,
    ) -> crate::Result<EmbeddingsResponse, EmbeddingsError> {
        match self
            .client
            .post(
                "/embeddings",
                json!({
                    "input": request.input,
                    "model": request.model,
                }),
            )
            .await
        {
            Ok(res) => Ok(res),
            Err(e) => Err(EmbeddingsError::ClientError(e)),
        }
    }
}

#[derive(Clone, Debug)]
pub struct GoogleConfig {
    pub api_config: ApiConfig,
    pub logging_config: LoggingConfig,
    pub extra_headers: HeaderMap,
}

impl Default for GoogleConfig {
    fn default() -> Self {
        Self {
            api_config: ApiConfig {
                host: GOOGLE_API_BASE.to_string(),
                port: None,
                api_key: None,
                api_key_env_var: "GEMINI_API_KEY".to_string(),
            },
            logging_config: LoggingConfig {
                logger_name: "gemini".to_string(),
                ..Default::default()
            },
            extra_headers: Default::default(),
        }
    }
}

impl GoogleConfig {
    pub fn new() -> Self {
        Default::default()
    }
}

impl ApiConfigTrait for GoogleConfig {
    fn headers(&self) -> HeaderMap {
        let mut headers = HeaderMap::new();

        if let Some(api_key) = self.api_key() {
            if let Ok(header_value) =
                HeaderValue::from_str(&format!("Bearer {}", api_key.expose_secret()))
            {
                headers.insert(AUTHORIZATION, header_value);
            } else {
                crate::error!("Failed to create header from authorization value");
            }
        }

        for (k, v) in &self.extra_headers {
            headers.insert(k.clone(), v.clone());
        }

        headers
    }

    fn url(&self, path: &str) -> String {
        if self.api_config.host.starts_with("http") {
            if let Some(port) = &self.api_config.port {
                format!("{}:{}{}", self.api_config.host, port, path)
            } else {
                format!("{}{}", self.api_config.host, path)
            }
        } else {
            format!("https://{}{}", self.api_config.host, path)
        }
    }

    fn api_key(&self) -> &Option<SecretString> {
        &self.api_config.api_key
    }
}
