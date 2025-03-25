use std::path::Path;

use super::{GraphOptimizationLevel, Session, TensorRef, inputs};
use alith_core::{
    chat::{Completion, CompletionError, Request, Response, TokenUsage},
    interface::{CompletionFinishReason, GenerationSettings, TimingUsage},
};
use alith_models::Tokenizer;
use anyhow::Result;
use rand::Rng;

/// Max tokens to generate
const GEN_TOKENS: usize = 90;
/// Top_K -> Sample from the k most likely next tokens at each step. Lower k focuses on higher probability tokens.
const TOP_K: usize = 5;

pub struct GPT2 {
    session: Session,
    tokenizer: Tokenizer,
}

impl GPT2 {
    /// Construct a GPT-2 model
    pub fn new(
        model_url: impl AsRef<str>,
        tokenizer_path: impl AsRef<Path>,
        opt_level: GraphOptimizationLevel,
        num_threads: usize,
    ) -> Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(opt_level)?
            .with_intra_threads(num_threads)?
            .commit_from_url(model_url)?;
        let tokenizer_path = tokenizer_path.as_ref();
        let tokenizer = if tokenizer_path.is_file() {
            Tokenizer::new_from_tokenizer_json(tokenizer_path)?
        } else {
            Tokenizer::new_from_hf_repo(None, tokenizer_path.as_os_str().to_str().unwrap())?
        };
        Ok(Self { session, tokenizer })
    }
}

impl Completion for GPT2 {
    type Response = Response;

    async fn completion(&mut self, request: Request) -> Result<Self::Response, CompletionError> {
        let finish_reason = CompletionFinishReason::Eos;
        let start_time = std::time::Instant::now();
        let tokens = self.tokenizer.tokenize(request.effective_prompt());
        let mut tokens = tokens.iter().map(|i| *i as i64).collect::<Vec<_>>();
        let prompt_tokens = tokens.len();
        let max_tokens = request.max_tokens.unwrap_or(GEN_TOKENS);
        let mut output = String::new();
        let mut rng = rand::rng();
        let top_k = request.top_k.unwrap_or(TOP_K);
        for _ in 0..max_tokens {
            // Raw tensor construction takes a tuple of (shape, data).
            // The model expects our input to have shape [B, _, S]
            let input =
                TensorRef::from_array_view((vec![1, 1, tokens.len() as i64], tokens.as_slice()))
                    .map_err(|err| CompletionError::Inference(err.to_string()))?;
            let outputs = self
                .session
                .run(inputs![input])
                .map_err(|err| CompletionError::Inference(err.to_string()))?;
            let (dim, mut probabilities) = outputs["output1"]
                .try_extract_tensor()
                .map_err(|err| CompletionError::Inference(err.to_string()))?;
            // The output tensor will have shape [B, _, S, V]
            // We want only the probabilities for the last token in this sequence, which will be the next most likely token
            // according to the model
            let (seq_len, vocab_size) = (dim[2] as usize, dim[3] as usize);
            probabilities = &probabilities[(seq_len - 1) * vocab_size..];

            // Sort each token by probability
            let mut probabilities: Vec<(usize, f32)> =
                probabilities.iter().copied().enumerate().collect();
            probabilities
                .sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Less));

            // Sample using top-k sampling
            let token = probabilities[rng.random_range(0..=top_k)].0 as i64;

            // Add our generated token to the input sequence
            tokens.push(token);

            let token_str = self
                .tokenizer
                .detokenize_many(&[token as u32])
                .map_err(|err| CompletionError::Inference(err.to_string()))?;
            output.push_str(&token_str);
        }
        Ok(Response {
            id: "GPT-2".to_string(),
            index: None,
            content: output,
            finish_reason,
            completion_probabilities: None,
            truncated: false,
            generation_settings: GenerationSettings::default(),
            timing_usage: TimingUsage::new_from_generic(start_time),
            token_usage: TokenUsage {
                tokens_cached: None,
                prompt_tokens: prompt_tokens as u32,
                completion_tokens: max_tokens as u32,
                total_tokens: (prompt_tokens + max_tokens) as u32,
                dollar_cost: None,
                cents_cost: None,
            },
            tool_calls: None,
        })
    }
}
