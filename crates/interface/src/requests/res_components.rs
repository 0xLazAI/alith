use super::completion::request::CompletionRequest;
use crate::llms::api::{
    anthropic::completion::AnthropicCompletionResponse,
    openai::completion::OpenAICompletionResponse,
};

/// The log probability of the completion.
#[derive(Debug)]
pub struct InferenceProbabilities {
    /// The token selected by the model.
    pub content: Option<String>,
    /// An array of length n_probs.
    pub top_probs: Vec<TopProbabilities>,
}

#[derive(Debug)]
pub struct TopProbabilities {
    /// The token.
    pub token: String,
    /// The log probability of this token.
    pub prob: f32,
}

/// The settings used to generate the completion.
#[derive(Default)]
pub struct GenerationSettings {
    /// The model used
    pub model: String,
    // pub prompt: String, // Need to think how to handle tokens vs. text
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: f32,
    pub temperature: f32,
    pub top_p: Option<f32>,
    /// The number of choices to generate.
    pub n_choices: u8,
    /// The number of tokens to predict same as max_tokens.
    pub n_predict: Option<i32>,
    /// The maxium context size of the model or server setting.
    pub n_ctx: u64,
    pub logit_bias: Option<Vec<Vec<serde_json::Value>>>,
    pub grammar: Option<String>,
    pub stop_sequences: Vec<String>, // change toi vec of stop sequences
}

impl GenerationSettings {
    pub fn new_from_model(model: &str) -> Self {
        Self {
            model: model.to_owned(),
            ..Default::default()
        }
    }

    pub fn new_from_openai(req: &CompletionRequest, res: &OpenAICompletionResponse) -> Self {
        Self {
            model: res.model.to_owned(),
            frequency_penalty: req.config.frequency_penalty,
            presence_penalty: req.config.presence_penalty,
            temperature: req.config.temperature,
            top_p: req.config.top_p,
            n_choices: 1,
            n_predict: req.config.actual_request_tokens.map(|x| x as i32),
            n_ctx: req.config.inference_ctx_size,
            logit_bias: None,
            grammar: None,
            stop_sequences: req
                .stop_sequences
                .sequences
                .iter()
                .map(|x| x.as_str().to_owned())
                .collect(),
        }
    }

    pub fn new_from_anthropic(req: &CompletionRequest, res: &AnthropicCompletionResponse) -> Self {
        Self {
            model: res.model.to_string(),
            frequency_penalty: req.config.frequency_penalty,
            presence_penalty: req.config.presence_penalty,
            temperature: req.config.temperature,
            top_p: req.config.top_p,
            n_choices: 1,
            n_predict: req.config.actual_request_tokens.map(|x| x as i32),
            n_ctx: req.config.inference_ctx_size,
            logit_bias: None,
            grammar: None,
            stop_sequences: req
                .stop_sequences
                .sequences
                .iter()
                .map(|x| x.as_str().to_owned())
                .collect(),
        }
    }
}

impl std::fmt::Display for GenerationSettings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f)?;
        writeln!(f, "    model: {:?}", self.model)?;
        writeln!(f, "    frequency_penalty: {:?}", self.frequency_penalty)?;
        writeln!(f, "    presence_penalty: {:?}", self.presence_penalty)?;
        writeln!(f, "    temperature: {:?}", self.temperature)?;
        writeln!(f, "    top_p: {:?}", self.top_p)?;
        writeln!(f, "    n_choices: {:?}", self.n_choices)?;
        writeln!(f, "    n_predict: {:?}", self.n_predict)?;
        writeln!(f, "    n_ctx: {:?}", self.n_ctx)?;
        writeln!(f, "    logit_bias: {:?}", self.logit_bias)?;
        writeln!(f, "    grammar: {:?}", self.grammar)?;
        writeln!(f, "    stop_sequences: {:?}", self.stop_sequences)
    }
}

/// Timing statistics for the completion request.
pub struct TimingUsage {
    /// Timestamp of when the request was created.
    pub start_time: std::time::Instant,
    /// Timestamp of when the request was completed.
    pub end_time: std::time::Instant,
    /// Total time taken to process the request.
    pub total_time: std::time::Duration,
    /// Time taken to process the prompt.
    pub prompt_processing_t: Option<std::time::Duration>,
    /// Time taken to generate the completion.
    pub generation_t: Option<std::time::Duration>,
    /// Number of prompt tokens processed per millisecond.
    pub prompt_tok_per_ms: Option<f32>,
    /// Number of prompt tokens processed per second.
    pub prompt_tok_per_sec: Option<f32>,
    /// Number of tokens generated per millisecond.
    pub generation_tok_per_ms: Option<f32>,
    /// Number of tokens generated per second.
    pub generation_tok_per_sec: Option<f32>,
}

impl TimingUsage {
    pub fn new_from_generic(start_time: std::time::Instant) -> Self {
        Self {
            total_time: start_time.elapsed(),
            start_time,
            end_time: std::time::Instant::now(),
            prompt_processing_t: None,
            generation_t: None,
            prompt_tok_per_ms: None,
            prompt_tok_per_sec: None,
            generation_tok_per_ms: None,
            generation_tok_per_sec: None,
        }
    }
}

impl std::fmt::Display for TimingUsage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f)?;
        writeln!(f, "    total_time: {:?}", self.total_time)?;
        writeln!(f, "    prompt_processing_t: {:?}", self.prompt_processing_t)?;
        writeln!(f, "    generation_t: {:?}", self.generation_t)?;
        writeln!(f, "    prompt_tok_per_ms: {:?}", self.prompt_tok_per_ms)?;
        writeln!(f, "    prompt_tok_per_sec: {:?}", self.prompt_tok_per_sec)?;
        writeln!(
            f,
            "    generation_tok_per_ms: {:?}",
            self.generation_tok_per_ms
        )?;
        writeln!(
            f,
            "    generation_tok_per_sec: {:?}",
            self.generation_tok_per_sec
        )
    }
}

/// Token statistics for the completion request.
#[derive(Debug, Clone)]
pub struct TokenUsage {
    /// Number of tokens from the prompt which could be re-used from previous completion (n_past)
    pub tokens_cached: Option<u32>,
    /// Number of tokens evaluated in total from the prompt. Same as tokens_evaluated.
    pub prompt_tokens: u32,
    /// Number of tokens in the generated completion. Same as predicted_n.
    pub completion_tokens: u32,
    /// Total number of tokens used in the request (prompt + completion).
    pub total_tokens: u32,
}

impl TokenUsage {
    pub fn new_from_generic(res: &OpenAICompletionResponse) -> Self {
        if let Some(usage) = &res.usage {
            Self {
                tokens_cached: None,
                prompt_tokens: usage.prompt_tokens,
                completion_tokens: usage.completion_tokens,
                total_tokens: usage.total_tokens,
            }
        } else {
            Self {
                tokens_cached: None,
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
            }
        }
    }

    pub fn new_from_anthropic(res: &AnthropicCompletionResponse) -> Self {
        Self {
            tokens_cached: None,
            prompt_tokens: res.usage.input_tokens,
            completion_tokens: res.usage.output_tokens,
            total_tokens: res.usage.input_tokens + res.usage.output_tokens,
        }
    }
}

impl std::fmt::Display for TokenUsage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f)?;
        writeln!(f, "    tokens_cached: {:?}", self.tokens_cached)?;
        writeln!(f, "    prompt_tokens: {:?}", self.prompt_tokens)?;
        writeln!(f, "    completion_tokens: {:?}", self.completion_tokens)?;
        writeln!(f, "    total_tokens: {:?}", self.total_tokens)
    }
}
