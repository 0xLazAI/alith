pub mod builder;
pub mod completion;

use super::{
    client::ApiClient,
    config::{ApiConfig, ApiConfigTrait},
};
use crate::requests::{
    completion::{
        error::CompletionError, request::CompletionRequest, response::CompletionResponse,
    },
    embeddings::{EmbeddingsError, EmbeddingsRequest, EmbeddingsResponse},
};
use alith_devices::logging::LoggingConfig;
use alith_models::api_model::ApiLLMModel;
use completion::OpenAICompletionRequest;
use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderValue};
use secrecy::{ExposeSecret, SecretString};
use serde_json::json;

/// Default v1 API base url
pub const OPENAI_API_HOST: &str = "api.openai.com/v1";
/// Organization header
pub const OPENAI_ORGANIZATION_HEADER: &str = "OpenAI-Organization";
/// Project header
pub const OPENAI_PROJECT_HEADER: &str = "OpenAI-Project";

pub struct OpenAIBackend {
    pub(crate) client: ApiClient<OpenAIConfig>,
    pub model: ApiLLMModel,
}

impl OpenAIBackend {
    pub fn new(mut config: OpenAIConfig, model: ApiLLMModel) -> crate::Result<Self> {
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
pub struct OpenAIConfig {
    pub api_config: ApiConfig,
    pub logging_config: LoggingConfig,
    pub org_id: String,
    pub project_id: String,
    pub extra_headers: HeaderMap,
}

impl Default for OpenAIConfig {
    fn default() -> Self {
        Self {
            api_config: ApiConfig {
                host: OPENAI_API_HOST.to_string(),
                port: None,
                api_key: None,
                api_key_env_var: "OPENAI_API_KEY".to_string(),
            },
            logging_config: LoggingConfig {
                logger_name: "openai".to_string(),
                ..Default::default()
            },
            org_id: Default::default(),
            project_id: Default::default(),
            extra_headers: Default::default(),
        }
    }
}

impl OpenAIConfig {
    pub fn new() -> Self {
        Default::default()
    }

    /// To use a different organization id other than default
    pub fn with_org_id<S: Into<String>>(mut self, org_id: S) -> Self {
        self.org_id = org_id.into();
        self
    }

    /// Non default project id
    pub fn with_project_id<S: Into<String>>(mut self, project_id: S) -> Self {
        self.project_id = project_id.into();
        self
    }
}

impl ApiConfigTrait for OpenAIConfig {
    fn headers(&self) -> HeaderMap {
        let mut headers = HeaderMap::new();

        if !self.org_id.is_empty() {
            if let Ok(header_value) = HeaderValue::from_str(self.org_id.as_str()) {
                headers.insert(OPENAI_ORGANIZATION_HEADER, header_value);
            } else {
                crate::error!("Failed to create header value from org_id value");
            }
        }
        if !self.project_id.is_empty() {
            if let Ok(header_value) = HeaderValue::from_str(self.project_id.as_str()) {
                headers.insert(OPENAI_PROJECT_HEADER, header_value);
            } else {
                crate::error!("Failed to create header value from project_id value");
            }
        }
        if let Some(api_key) = self.api_key() {
            if let Ok(header_value) =
                HeaderValue::from_str(&format!("Bearer {}", api_key.expose_secret()))
            {
                headers.insert(AUTHORIZATION, header_value);
            } else {
                crate::error!("Failed to create header value from authorization value");
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
