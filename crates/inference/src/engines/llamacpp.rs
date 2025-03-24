use std::{
    num::NonZeroU32,
    path::Path,
    sync::{Mutex, OnceLock},
};

use crate::errors::InferenceError;
use alith_core::{
    chat::{Completion, CompletionError, Request as CompletionRequest},
    client::CompletionResponse,
    interface::requests::completion::{
        CompletionFinishReason, GenerationSettings, TimingUsage, TokenUsage,
    },
};
use alith_models::local_model::{GgufLoader, GgufLoaderTrait, LocalLLMModel};
use llama_cpp_2::{
    LLamaCppError,
    context::{LlamaContext, params::LlamaContextParams},
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{AddBos, LlamaModel, Special, params::LlamaModelParams},
    sampling::LlamaSampler,
};
use tokio::sync::OnceCell;

const DEFAULT_MAX_TOKENS: usize = 8192;
const CONTEXT_SIZE: u32 = 8192;

static LLAMA_BACKEND: OnceCell<LlamaBackend> = OnceCell::const_new();
pub(crate) static LLAMA_MODEL: OnceCell<LlamaModel> = OnceCell::const_new();
const NUM_CONTEXTS: usize = 3;
static LLAMA_CONTEXTS: [OnceLock<Mutex<ContextWrapper>>; NUM_CONTEXTS] =
    [OnceLock::new(), OnceLock::new(), OnceLock::new()];

// Newtype to simplify LlamaContext lifetime
#[derive(Debug)]
struct ContextWrapper(LlamaContext<'static>);
unsafe impl Send for ContextWrapper {} // LlamaContext has a NonNull which is !Send
unsafe impl Sync for ContextWrapper {} // LlamaContext has a NonNull which is !Sync

pub struct LlamaEngine {
    model: LocalLLMModel,
}

impl LlamaEngine {
    pub async fn new<P: AsRef<Path>>(model_path: P) -> Result<Self, InferenceError> {
        let backend = LlamaBackend::init()?;
        let model = load_model(&backend, model_path.as_ref())?;
        LLAMA_MODEL
            .set(model)
            .map_err(|err| InferenceError::General(err.to_string()))?;
        // Safety: NonZeroU32::new only errors if we give it a zero
        let context_size = NonZeroU32::new(CONTEXT_SIZE).unwrap();
        let llama_ctx_params = LlamaContextParams::default().with_n_ctx(Some(context_size));
        for (_, ctx_holder) in LLAMA_CONTEXTS.iter().enumerate().take(NUM_CONTEXTS) {
            let llama_ctx = LLAMA_MODEL
                .get()
                .unwrap() // Safety: We put it in a few lines up
                .new_context(&backend, llama_ctx_params.clone())
                .map_err(LLamaCppError::LlamaContextLoadError)?;
            let _ = ctx_holder.set(Mutex::new(ContextWrapper(llama_ctx)));
        }
        LLAMA_BACKEND
            .set(backend)
            .map_err(|err| InferenceError::General(err.to_string()))?;
        Ok(LlamaEngine {
            model: GgufLoader::default()
                .local_quant_file_path(model_path.as_ref())
                .load()
                .map_err(|err| InferenceError::General(err.to_string()))?,
        })
    }
}

fn load_model<P: AsRef<Path>>(
    backend: &LlamaBackend,
    path: P,
) -> Result<LlamaModel, InferenceError> {
    let model_params = {
        if cfg!(any(feature = "cuda", feature = "vulkan")) {
            LlamaModelParams::default().with_n_gpu_layers(1000)
        } else {
            LlamaModelParams::default()
        }
    };
    LlamaModel::load_from_file(backend, path, &model_params)
        .map_err(|err| InferenceError::ModelLoad(err.to_string()))
}

impl Completion for LlamaEngine {
    type Response = CompletionResponse;

    async fn completion(
        &mut self,
        request: CompletionRequest,
    ) -> Result<Self::Response, CompletionError> {
        let model = LLAMA_MODEL.get().unwrap();
        let mut llama_context = LLAMA_CONTEXTS[0].get().unwrap().lock().unwrap();
        let input = self
            .model
            .chat_template
            .apply(&request.map_messages(), true);
        let tokens_list = model
            .str_to_token(&input, AddBos::Always)
            .map_err(|err| CompletionError::Inference(err.to_string()))?;
        let limit = DEFAULT_MAX_TOKENS; // - prompt_tokens;
        let max_output_tokens = std::cmp::min(request.max_tokens.unwrap_or(limit), limit);
        // Create a llama_batch with size 512
        // we use this object to submit token data for decoding
        let mut batch = LlamaBatch::new(512, 1);
        let last_index: i32 = (tokens_list.len() - 1) as i32;
        for (i, token) in (0_i32..).zip(tokens_list.into_iter()) {
            // llama_decode will output logits only for the last token of the prompt
            let is_last = i == last_index;
            batch
                .add(token, i, &[0], is_last)
                .map_err(|err| CompletionError::Inference(err.to_string()))?;
        }

        // "decode" means "run forward pass"
        llama_context
            .0
            .decode(&mut batch)
            .map_err(|err| CompletionError::Inference(err.to_string()))?;

        let mut sampler = LlamaSampler::greedy();
        if let Some(temperature) = request.temperature {
            sampler = LlamaSampler::chain([sampler, LlamaSampler::temp(temperature)], false);
        }
        if let Some(top_p) = request.top_p {
            sampler = LlamaSampler::chain([sampler, LlamaSampler::top_p(top_p, 1)], false);
        }
        if let Some(top_k) = request.top_k {
            sampler = LlamaSampler::chain([sampler, LlamaSampler::top_k(top_k as i32)], false)
        }

        let mut n_cur = batch.n_tokens() as u32;

        let mut output_tokens = Vec::with_capacity(max_output_tokens);
        let mut output = String::new();

        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut finish_reason = CompletionFinishReason::Eos;
        let start_time = std::time::Instant::now();

        loop {
            // Sample the next token
            let n_tokens = batch.n_tokens();
            let token = sampler.sample(&llama_context.0, n_tokens - 1);
            sampler.accept(token);
            // Is it an end of stream?
            // This is probably safe for concurrent access
            if model.is_eog_token(token) {
                break;
            }
            let output_bytes = model.token_to_bytes(token, Special::Tokenize);
            if let Ok(output_bytes) = output_bytes {
                // Use `Decoder.decode_to_string()` to avoid the intermediate buffer
                let mut output_string = String::with_capacity(32);
                let _decode_result =
                    decoder.decode_to_string(&output_bytes, &mut output_string, false);
                output.push_str(&output_string);
            };
            output_tokens.push(token.0 as u32);
            batch.clear();
            if let Err(err) = batch.add(token, n_cur as i32, &[0], true) {
                return Err(CompletionError::Inference(format!(
                    "batch add error, probably insufficient space in buffer, aborting request. {err}."
                )));
            }
            n_cur += 1;

            if output_tokens.len() > max_output_tokens {
                finish_reason = CompletionFinishReason::StopLimit;
                break;
            }

            llama_context
                .0
                .decode(&mut batch)
                .map_err(|err| CompletionError::Inference(err.to_string()))?;
        }
        Ok(CompletionResponse {
            id: self.model.model_base.model_id.clone(),
            index: None,
            content: output,
            finish_reason,
            completion_probabilities: None,
            truncated: false,
            generation_settings: GenerationSettings::default(),
            timing_usage: TimingUsage::new_from_generic(start_time),
            token_usage: TokenUsage {
                tokens_cached: None,
                prompt_tokens: (last_index + 1) as u32,
                completion_tokens: output_tokens.len() as u32,
                total_tokens: (last_index as usize + 1 + output_tokens.len()) as u32,
                dollar_cost: None,
                cents_cost: None,
            },
            // TODO: tool calls
            tool_calls: None,
        })
    }
}
