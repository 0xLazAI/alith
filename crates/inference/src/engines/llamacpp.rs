use anyhow::Context;
use std::{
    num::NonZeroU32,
    path::Path,
    sync::{Mutex, OnceLock},
};

use crate::errors::InferenceError;
use llama_cpp_2::{
    LLamaCppError,
    context::{LlamaContext, params::LlamaContextParams},
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{LlamaModel, params::LlamaModelParams},
    sampling::LlamaSampler,
    token::LlamaToken,
};
use tokio::sync::OnceCell;
pub use tokio_util::sync::CancellationToken;

use super::{Annotated, EngineOutput, Request};

const DEFAULT_MAX_TOKENS: u32 = 8192;
const CONTEXT_SIZE: u32 = 8192;

static LLAMA_BACKEND: OnceCell<LlamaBackend> = OnceCell::const_new();
pub(crate) static LLAMA_MODEL: OnceCell<LlamaModel> = OnceCell::const_new();
const NUM_CONTEXTS: usize = 3;
static LLAMA_CONTEXTS: [OnceLock<Mutex<ContextWrapper>>; NUM_CONTEXTS] =
    [OnceLock::new(), OnceLock::new(), OnceLock::new()];

struct WorkRequest {
    request: Request,
    response_channel: tokio::sync::mpsc::Sender<Annotated<EngineOutput>>,
}

struct LlamacppEngine {
    cancel_token: CancellationToken,
    req_tx: tokio::sync::mpsc::Sender<WorkRequest>,
}

// Newtype to simplify LlamaContext lifetime
#[derive(Debug)]
struct ContextWrapper(LlamaContext<'static>);
unsafe impl Send for ContextWrapper {} // LlamaContext has a NonNull which is !Send
unsafe impl Sync for ContextWrapper {} // LlamaContext has a NonNull which is !Sync

pub struct LlamaEngine {
    cancel_token: CancellationToken,
    req_tx: tokio::sync::mpsc::Sender<WorkRequest>,
}

impl LlamaEngine {
    async fn new<P: AsRef<Path>>(
        cancel_token: CancellationToken,
        model_path: P,
    ) -> Result<Self, InferenceError> {
        let backend = LlamaBackend::init()?;
        let model = load_model(&backend, model_path)?;
        LLAMA_MODEL
            .set(model)
            .map_err(|err| InferenceError::General(err.to_string()))?;

        let (ctx_set, ctx_get) = tokio::sync::mpsc::channel(NUM_CONTEXTS);
        // Safety: NonZeroU32::new only errors if we give it a zero
        let context_size = NonZeroU32::new(CONTEXT_SIZE).unwrap();
        let llama_ctx_params = LlamaContextParams::default().with_n_ctx(Some(context_size));
        for (i, ctx_holder) in LLAMA_CONTEXTS.iter().enumerate().take(NUM_CONTEXTS) {
            let llama_ctx = LLAMA_MODEL
                .get()
                .unwrap() // Safety: We put it in a few lines up
                .new_context(&backend, llama_ctx_params.clone())
                .map_err(LLamaCppError::LlamaContextLoadError)?;
            let _ = ctx_holder.set(Mutex::new(ContextWrapper(llama_ctx)));
            let _ = ctx_set.send(i).await;
        }
        LLAMA_BACKEND
            .set(backend)
            .map_err(|err| InferenceError::General(err.to_string()))?;
        let (req_tx, req_rx) = tokio::sync::mpsc::channel(2);
        let ct = cancel_token.clone();
        tokio::task::spawn(worker(ct, req_rx, ctx_get, ctx_set));
        Ok(LlamaEngine {
            cancel_token,
            req_tx,
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

// Run this in a thread
async fn worker(
    cancel_token: CancellationToken,
    mut req_rx: tokio::sync::mpsc::Receiver<WorkRequest>,
    mut ctx_get: tokio::sync::mpsc::Receiver<usize>,
    ctx_set: tokio::sync::mpsc::Sender<usize>,
) {
    loop {
        let maybe_work_request = tokio::select! {
            _ = cancel_token.cancelled() => {
                break;
            }
            maybe_work_request = req_rx.recv() => {
                maybe_work_request
            }
        };
        let Some(work_request) = maybe_work_request else {
            break;
        };
        // will block if there are already NUM_CONTEXTS requests in flight
        let Some(ctx_pos) = ctx_get.recv().await else {
            unreachable!("We don't close ctx_set");
        };
        let ct = cancel_token.clone();
        let inner_ctx_set = ctx_set.clone();

        tokio::task::spawn_blocking(move || {
            let mut ctx = LLAMA_CONTEXTS[ctx_pos].get().unwrap().lock().unwrap();
            if let Err(err) = run_request(ct, work_request, &mut ctx) {
                eprintln!("{err}");
            }
            let _ = inner_ctx_set.blocking_send(ctx_pos);
        });
    }
}

fn run_request(
    cancel_token: CancellationToken,
    work_request: WorkRequest,
    llama_context: &mut ContextWrapper,
) -> anyhow::Result<()> {
    let tokens_list: Vec<LlamaToken> = work_request
        .request
        .token_ids
        .into_iter()
        .map(|u| LlamaToken::new(u as i32))
        .collect();

    let limit = DEFAULT_MAX_TOKENS; // - prompt_tokens;
    let max_output_tokens = std::cmp::min(work_request.request.max_tokens.unwrap_or(limit), limit);

    // create a llama_batch with size 512
    // we use this object to submit token data for decoding
    let mut batch = LlamaBatch::new(512, 1);
    let last_index: i32 = (tokens_list.len() - 1) as i32;
    for (i, token) in (0_i32..).zip(tokens_list.into_iter()) {
        // llama_decode will output logits only for the last token of the prompt
        let is_last = i == last_index;
        batch
            .add(token, i, &[0], is_last)
            .with_context(|| format!("Failed adding token pos {i} to batch"))?;
    }

    // "decode" means "run forward pass"
    llama_context
        .0
        .decode(&mut batch)
        .with_context(|| "llama_decode failed on first pass")?;

    let mut sampler = LlamaSampler::greedy();
    let mut n_cur = batch.n_tokens() as u32;

    let mut used_output_tokens = 0;
    while !cancel_token.is_cancelled() {
        // sample the next token
        let n_tokens = batch.n_tokens();
        let token = sampler.sample(&llama_context.0, n_tokens - 1);
        sampler.accept(token);

        // is it an end of stream?
        // This is probably safe for concurrent access
        if LLAMA_MODEL.get().unwrap().is_eog_token(token) {
            work_request
                .response_channel
                .blocking_send(Annotated::from_data(EngineOutput::stop()))
                .with_context(|| "Failed sending stop to response_channel")?;
            break;
        }

        let engine_out = EngineOutput {
            // todo - propagate mdcsum
            token_ids: vec![token.0 as u32],
            tokens: None,
            text: None,
            finish_reason: None,
        };
        work_request
            .response_channel
            .blocking_send(Annotated::from_data(engine_out))
            .with_context(|| "Failed forwarding engine output to response_channel")?;

        batch.clear();
        if let Err(err) = batch.add(token, n_cur as i32, &[0], true) {
            let err_msg = format!(
                "batch add error, probably insufficient space in buffer, aborting request. {err}."
            );
            eprintln!("{err_msg}");
            let _ = work_request
                .response_channel
                .blocking_send(Annotated::from_data(EngineOutput::error(err_msg)));
            break;
        }
        n_cur += 1;

        used_output_tokens += 1;
        if used_output_tokens > max_output_tokens {
            let _ = work_request
                .response_channel
                .blocking_send(Annotated::from_data(EngineOutput::length()));
            break;
        }

        llama_context
            .0
            .decode(&mut batch)
            .with_context(|| "llama_decode failed during loop")?;
    }
    if cancel_token.is_cancelled() {
        let _ = work_request
            .response_channel
            .blocking_send(Annotated::from_data(EngineOutput::stop()));
    }

    // Clean context for next use
    llama_context.0.clear_kv_cache();
    llama_context.0.reset_timings();

    Ok(())
}
