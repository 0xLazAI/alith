use alith_interface::{
    llms::LLMBackend,
    requests::{
        completion::{CompletionRequest, CompletionResponse},
        logit_bias::{LogitBias, LogitBiasTrait},
        req_components::{RequestConfig, RequestConfigTrait},
    },
};
use alith_prompt::LLMPrompt;
use std::sync::Arc;

pub type ChatCompletion = BasicCompletion;

#[derive(Clone)]
pub struct BasicCompletion {
    pub base_req: CompletionRequest,
}

impl BasicCompletion {
    #[inline]
    pub fn new(backend: Arc<LLMBackend>) -> Self {
        Self {
            base_req: CompletionRequest::new(backend),
        }
    }

    #[inline]
    pub fn prompt(&mut self) -> &mut LLMPrompt {
        &mut self.base_req.prompt
    }

    #[inline]
    pub async fn run(&mut self) -> crate::Result<CompletionResponse> {
        Ok(self.base_req.request().await?)
    }

    pub fn parse_response(&self, content: &str) -> crate::Result<String> {
        if content.is_empty() {
            return Err(anyhow::format_err!(
                "parse_response error: content.is_empty()"
            ));
        }

        let content = content
            .strip_prefix("assistant\n\n")
            .or_else(|| content.strip_prefix("assistant\n"))
            .or_else(|| content.strip_prefix("assistant"))
            .unwrap_or(content);

        Ok(content.trim().to_owned())
    }
}

impl RequestConfigTrait for BasicCompletion {
    fn config(&mut self) -> &mut RequestConfig {
        &mut self.base_req.config
    }

    fn reset_request(&mut self) {
        self.base_req.reset_completion_request();
    }
}

impl LogitBiasTrait for BasicCompletion {
    fn lb_mut(&mut self) -> &mut Option<LogitBias> {
        &mut self.base_req.logit_bias
    }
}
