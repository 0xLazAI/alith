use crate::{llms::LLMBackend, requests::req_components::RequestConfig};
use std::sync::Arc;

use super::{EmbeddingsError, response::EmbeddingsResponse};

pub struct EmbeddingsRequest {
    pub model: String,
    pub input: Vec<String>,
    backend: Arc<LLMBackend>,
    config: RequestConfig,
    llm_interface_errors: Vec<EmbeddingsError>,
}

impl Clone for EmbeddingsRequest {
    fn clone(&self) -> Self {
        Self {
            model: self.model.clone(),
            input: self.input.clone(),
            backend: self.backend.clone(),
            config: self.config.clone(),
            llm_interface_errors: Vec::new(),
        }
    }
}

impl EmbeddingsRequest {
    pub fn new(backend: Arc<LLMBackend>) -> EmbeddingsRequest {
        EmbeddingsRequest {
            model: String::new(),
            input: Vec::new(),
            backend: Arc::clone(&backend),
            config: RequestConfig::new(backend.model_ctx_size(), backend.inference_ctx_size()),
            llm_interface_errors: Vec::new(),
        }
    }

    pub fn reset_embedding_request(&mut self) {
        self.input = Vec::new();
    }

    pub async fn request(&mut self) -> crate::Result<EmbeddingsResponse, EmbeddingsError> {
        self.llm_interface_errors.clear();
        let mut retry_count: u8 = 0;
        loop {
            if retry_count >= self.config.retry_after_fail_n_times {
                let llm_interface_error = EmbeddingsError::ExceededRetryCount {
                    message: format!("Request failed after {retry_count} attempts."),
                    errors: std::mem::take(&mut self.llm_interface_errors),
                };
                tracing::error!(?llm_interface_error);
                eprintln!("{}", llm_interface_error);
                return Err(llm_interface_error);
            }
            tracing::info!("{}", self);
            match self.backend.embeddings_request(self).await {
                Err(e) => {
                    tracing::warn!(?e);
                    retry_count += 1;
                    match e {
                        EmbeddingsError::RequestBuilderError { .. }
                        | EmbeddingsError::ClientError { .. } => {
                            return Err(e);
                        }

                        _ => (),
                    }
                    self.llm_interface_errors.push(e);
                    continue;
                }
                Ok(res) => {
                    tracing::info!("{:?}", res);
                    return Ok(res);
                }
            };
        }
    }

    pub fn set_input(&mut self, input: Vec<String>) {
        self.input = input;
    }
}

impl std::fmt::Display for EmbeddingsRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f)?;
        writeln!(f, "CompletionRequest:")?;
        writeln!(f, "  input: {:?}", self.input)
    }
}
