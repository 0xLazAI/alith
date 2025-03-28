use super::{ToolChoice, ToolDefinition, error::CompletionError, response::CompletionResponse};
use crate::{
    llms::LLMBackend,
    requests::{
        completion::response::CompletionFinishReason, logit_bias::LogitBias,
        req_components::RequestConfig, stop_sequence::StopSequences,
    },
};
use alith_prompt::LLMPrompt;
use std::sync::Arc;

pub struct CompletionRequest {
    pub start_time: std::time::Instant,
    pub stop_sequences: StopSequences,
    pub grammar_string: Option<String>,
    pub logit_bias: Option<LogitBias>,
    pub prompt: LLMPrompt,
    pub config: RequestConfig,
    pub backend: Arc<LLMBackend>,
    pub llm_interface_errors: Vec<CompletionError>,
    pub tools: Vec<ToolDefinition>,
    pub tool_choice: ToolChoice,
}

impl Clone for CompletionRequest {
    fn clone(&self) -> Self {
        Self {
            start_time: self.start_time,
            stop_sequences: self.stop_sequences.clone(),
            grammar_string: self.grammar_string.clone(),
            logit_bias: self.logit_bias.clone(),
            prompt: self.prompt.clone(),
            config: self.config.clone(),
            backend: Arc::clone(&self.backend),
            llm_interface_errors: Vec::new(),
            tools: Vec::new(),
            tool_choice: ToolChoice::Auto,
        }
    }
}

impl CompletionRequest {
    pub fn new(backend: Arc<LLMBackend>) -> CompletionRequest {
        CompletionRequest {
            start_time: std::time::Instant::now(),
            stop_sequences: Default::default(),
            logit_bias: None,
            config: RequestConfig::new(backend.model_ctx_size(), backend.inference_ctx_size()),
            prompt: backend.new_prompt(),
            grammar_string: None,
            backend: Arc::clone(&backend),
            llm_interface_errors: Vec::new(),
            tools: Vec::new(),
            tool_choice: ToolChoice::default(),
        }
    }

    pub fn reset_completion_request(&mut self) {
        self.prompt.reset_prompt();
        self.stop_sequences.sequences.clear();
        self.grammar_string = None;
        self.logit_bias = None;
    }

    pub async fn request(&mut self) -> crate::Result<CompletionResponse, CompletionError> {
        self.llm_interface_errors.clear();
        self.start_time = std::time::Instant::now();
        self.backend
            .build_logit_bias(&mut self.logit_bias)
            .map_err(|e| CompletionError::RequestBuilderError(e.to_string()))?;

        let total_prompt_tokens = self
            .backend
            .get_total_prompt_tokens(&self.prompt)
            .map_err(|e| CompletionError::RequestBuilderError(e.to_string()))?;

        self.config
            .set_max_tokens_for_request(total_prompt_tokens as u64)
            .map_err(CompletionError::RequestTokenLimitError)?;

        let mut retry_count: u8 = 0;

        loop {
            if retry_count >= self.config.retry_after_fail_n_times {
                let llm_interface_error = CompletionError::ExceededRetryCount {
                    message: format!("Request failed after {retry_count} attempts."),
                    errors: std::mem::take(&mut self.llm_interface_errors),
                };
                tracing::error!(?llm_interface_error);
                eprintln!("{}", llm_interface_error);
                return Err(llm_interface_error);
            }
            tracing::info!("{}", self);
            match self.backend.completion_request(self).await {
                Err(e) => {
                    tracing::warn!(?e);
                    retry_count += 1;
                    match e {
                        CompletionError::RequestBuilderError { .. }
                        | CompletionError::StopReasonUnsupported { .. }
                        | CompletionError::ClientError { .. } => {
                            return Err(e);
                        }

                        _ => (),
                    }
                    self.llm_interface_errors.push(e);
                    continue;
                }
                Ok(res) => {
                    tracing::info!("{}", res);
                    if self.stop_sequences.required {
                        if matches!(
                            res.finish_reason,
                            CompletionFinishReason::MatchingStoppingSequence(_)
                        ) {
                            return Ok(res);
                        } else {
                            let llm_interface_error = match res.finish_reason {
                                CompletionFinishReason::NonMatchingStoppingSequence(s) => {
                                    if let Some(s) = s {
                                        CompletionError::NonMatchingStopSequence(s.clone())
                                    } else {
                                        CompletionError::NoRequiredStopSequence
                                    }
                                }
                                _ => CompletionError::NoRequiredStopSequence,
                            };
                            tracing::warn!(?llm_interface_error);
                            self.llm_interface_errors.push(llm_interface_error);
                            retry_count += 1;
                            if self.config.increase_limit_on_fail {
                                self.config
                                    .increase_token_limit(total_prompt_tokens as u64, None)?;
                            }
                            continue;
                        };
                    };
                    match res.finish_reason {
                        CompletionFinishReason::NonMatchingStoppingSequence(_)
                        | CompletionFinishReason::MatchingStoppingSequence(_) => return Ok(res),
                        CompletionFinishReason::StopLimit => {
                            if self.config.increase_limit_on_fail {
                                let llm_interface_error = CompletionError::StopLimitRetry;
                                tracing::warn!(?llm_interface_error);
                                self.llm_interface_errors.push(llm_interface_error);
                                self.config
                                    .increase_token_limit(total_prompt_tokens as u64, None)?;
                                retry_count += 1;
                                continue;
                            }
                            return Ok(res);
                        }
                        CompletionFinishReason::Eos | CompletionFinishReason::ToolsCall => {
                            return Ok(res);
                        }
                    }
                }
            };
        }
    }

    pub fn set_base_req_stop_sequences(
        &mut self,
        stop_word_done: &Option<String>,
        stop_word_no_result: &Option<String>,
    ) {
        if stop_word_done.is_some() || stop_word_no_result.is_some()
        // || step.stop_word_steps_done.is_some()
        {
            self.stop_sequences.required = true;
            self.stop_sequences.sequences.clear();
        }
        if let Some(stop_word_done) = &stop_word_done {
            self.stop_sequences.set_stop_word_done(stop_word_done);
        }

        if let Some(no_result_stop_word) = &stop_word_no_result {
            self.stop_sequences
                .set_stop_word_no_result(no_result_stop_word);
        }
    }
}

impl std::fmt::Display for CompletionRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f)?;
        writeln!(f, "CompletionRequest:")?;

        writeln!(
            f,
            "  total_prompt_tokens: {}",
            self.prompt.total_prompt_tokens()
        )?;
        write!(f, "  messages:\n\n{}", self.prompt.messages)?;
        writeln!(f, "  stop_sequences: {:?}", self.stop_sequences.to_vec())?;
        writeln!(f, "  grammar_string: {:?}", self.grammar_string)?;
        write!(f, "  config: {}", self.config)?;
        write!(f, "  tools: {:?}", self.tools)
    }
}
