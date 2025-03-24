use std::{num::NonZeroUsize, path::Path};

use anyhow::Result;

use alith_core::{
    chat::{Completion, CompletionError, Request as CompletionRequest},
    client::CompletionResponse,
    interface::requests::completion::{
        CompletionFinishReason, GenerationSettings, TimingUsage, TokenUsage,
    },
};
use anyhow::bail;
use mistralrs::{
    AutoDeviceMapParams, DefaultSchedulerMethod, DeviceMapSetting, Function, GGUFLoaderBuilder,
    GGUFSpecificConfig, MistralRsBuilder, Model, ModelDType, NormalLoaderBuilder,
    NormalSpecificConfig, PagedAttentionConfig, RequestBuilder, SchedulerConfig, TextMessageRole,
    TokenSource, Tool, ToolChoice, ToolType, Topology, best_device, initialize_logging,
    paged_attn_supported,
};

pub struct MistralRsEngine {
    model: Model,
}

impl MistralRsEngine {
    pub async fn new(model_id: impl ToString) -> Result<Self> {
        let builder = GgufModelBuilder::new(model_id).with_force_cpu();
        Ok(MistralRsEngine {
            model: builder.build().await?,
        })
    }
}

impl Completion for MistralRsEngine {
    type Response = CompletionResponse;

    async fn completion(
        &mut self,
        request: CompletionRequest,
    ) -> Result<Self::Response, CompletionError> {
        let mut messages = RequestBuilder::new();
        if let Some(temperature) = request.temperature {
            messages = messages.set_sampler_temperature(temperature as f64);
        }
        if let Some(max_tokens) = request.max_tokens {
            messages = messages.set_sampler_max_len(max_tokens);
        }
        if let Some(top_p) = request.top_p {
            messages = messages.set_sampler_topp(top_p as f64);
        }
        for m in &request.history {
            match m.role.as_str() {
                "system" => {
                    messages = messages.add_message(TextMessageRole::System, m.content.clone())
                }
                "user" => messages = messages.add_message(TextMessageRole::User, m.content.clone()),
                "assistant" => {
                    messages = messages.add_message(TextMessageRole::Assistant, m.content.clone())
                }
                _ => continue, // Just skip unknown roles
            };
        }
        messages = messages.add_message(TextMessageRole::User, request.effective_prompt());
        let mut tools = vec![];
        for tool in &request.tools {
            tools.push(Tool {
                tp: ToolType::Function,
                function: Function {
                    description: Some(tool.description.clone()),
                    name: tool.name.clone(),
                    parameters: serde_json::from_str(&serde_json::to_string(&tool.parameters)?)?,
                },
            });
        }
        if !tools.is_empty() {
            messages = messages.set_tools(tools);
            messages = messages.set_tool_choice(ToolChoice::Auto);
        }
        let start_time = std::time::Instant::now();
        let response = self
            .model
            .send_chat_request(messages.clone())
            .await
            .map_err(|err| CompletionError::Inference(err.to_string()))?;
        let choice = if response.choices.is_empty() {
            return Err(CompletionError::Inference(
                "ResponseContentEmpty: Response had no content".to_string(),
            ));
        } else {
            &response.choices[0]
        };
        let finish_reason = match choice.finish_reason.as_str() {
            "stop" | "canceled" => CompletionFinishReason::Eos,
            "length" => CompletionFinishReason::StopLimit,
            "tool_calls" => CompletionFinishReason::ToolsCall,
            _ => CompletionFinishReason::Eos,
        };
        let result = CompletionResponse {
            id: response.id,
            index: None,
            content: choice.message.content.as_ref().cloned().unwrap_or_default(),
            finish_reason,
            completion_probabilities: None,
            truncated: false,
            generation_settings: GenerationSettings::default(),
            timing_usage: TimingUsage::new_from_generic(start_time),
            token_usage: TokenUsage {
                tokens_cached: None,
                prompt_tokens: response.usage.prompt_tokens as u32,
                completion_tokens: response.usage.completion_tokens as u32,
                total_tokens: response.usage.total_tokens as u32,
                dollar_cost: None,
                cents_cost: None,
            },
            tool_calls: Some(
                choice
                    .message
                    .tool_calls
                    .iter()
                    .map(
                        |tool| alith_core::interface::requests::completion::tool::ToolCall {
                            id: tool.id.clone(),
                            r#type: tool.tp.to_string(),
                            function: alith_core::interface::requests::completion::tool::Function {
                                name: tool.function.name.clone(),
                                arguments: tool.function.arguments.clone(),
                            },
                        },
                    )
                    .collect(),
            ),
        };
        Ok(result)
    }
}

/// Configure a text GGUF model with the various parameters for loading, running, and other inference behaviors.
pub struct GgufModelBuilder {
    // Loading model
    pub(crate) model_id: String,
    pub(crate) tok_model_id: Option<String>,
    pub(crate) token_source: TokenSource,
    pub(crate) hf_revision: Option<String>,
    pub(crate) chat_template: Option<String>,
    pub(crate) tokenizer_json: Option<String>,
    pub(crate) device_mapping: Option<DeviceMapSetting>,

    // Model running
    pub(crate) prompt_chunksize: Option<NonZeroUsize>,
    pub(crate) force_cpu: bool,
    pub(crate) topology: Option<Topology>,

    // Other things
    pub(crate) paged_attn_cfg: Option<PagedAttentionConfig>,
    pub(crate) max_num_seqs: usize,
    pub(crate) no_kv_cache: bool,
    pub(crate) with_logging: bool,
    pub(crate) prefix_cache_n: Option<usize>,
}

impl GgufModelBuilder {
    /// A few defaults are applied here:
    /// - Token source is from the cache (.cache/huggingface/token)
    /// - Maximum number of sequences running is 32
    /// - Number of sequences to hold in prefix cache is 16.
    /// - Automatic device mapping with model defaults according to `AutoDeviceMapParams`
    pub fn new(model_id: impl ToString) -> Self {
        Self {
            model_id: model_id.to_string(),
            prompt_chunksize: None,
            chat_template: None,
            tokenizer_json: None,
            force_cpu: false,
            token_source: TokenSource::CacheToken,
            hf_revision: None,
            paged_attn_cfg: None,
            max_num_seqs: 32,
            no_kv_cache: false,
            prefix_cache_n: Some(16),
            with_logging: false,
            topology: None,
            tok_model_id: None,
            device_mapping: None,
        }
    }

    /// Source the tokenizer and chat template from this model ID (must contain `tokenizer.json` and `tokenizer_config.json`).
    pub fn with_tok_model_id(mut self, tok_model_id: impl ToString) -> Self {
        self.tok_model_id = Some(tok_model_id.to_string());
        self
    }

    /// Set the prompt batchsize to use for inference.
    pub fn with_prompt_chunksize(mut self, prompt_chunksize: NonZeroUsize) -> Self {
        self.prompt_chunksize = Some(prompt_chunksize);
        self
    }

    /// Set the model topology for use during loading. If there is an overlap, the topology type is used over the ISQ type.
    pub fn with_topology(mut self, topology: Topology) -> Self {
        self.topology = Some(topology);
        self
    }

    /// Literal Jinja chat template OR Path (ending in `.json`) to one.
    pub fn with_chat_template(mut self, chat_template: impl ToString) -> Self {
        self.chat_template = Some(chat_template.to_string());
        self
    }

    /// Path to a discrete `tokenizer.json` file.
    pub fn with_tokenizer_json(mut self, tokenizer_json: impl ToString) -> Self {
        self.tokenizer_json = Some(tokenizer_json.to_string());
        self
    }

    /// Force usage of the CPU device. Do not use PagedAttention with this.
    pub fn with_force_cpu(mut self) -> Self {
        self.force_cpu = true;
        self
    }

    /// Source of the Hugging Face token.
    pub fn with_token_source(mut self, token_source: TokenSource) -> Self {
        self.token_source = token_source;
        self
    }

    /// Set the revision to use for a Hugging Face remote model.
    pub fn with_hf_revision(mut self, revision: impl ToString) -> Self {
        self.hf_revision = Some(revision.to_string());
        self
    }

    /// Enable PagedAttention. Configure PagedAttention with a [`PagedAttentionConfig`] object, which
    /// can be created with sensible values with a [`PagedAttentionMetaBuilder`].
    ///
    /// If PagedAttention is not supported (query with [`paged_attn_supported`]), this will do nothing.
    ///
    /// [`PagedAttentionMetaBuilder`]: crate::PagedAttentionMetaBuilder
    pub fn with_paged_attn(
        mut self,
        paged_attn_cfg: impl FnOnce() -> anyhow::Result<PagedAttentionConfig>,
    ) -> anyhow::Result<Self> {
        if paged_attn_supported() {
            self.paged_attn_cfg = Some(paged_attn_cfg()?);
        } else {
            self.paged_attn_cfg = None;
        }
        Ok(self)
    }

    /// Set the maximum number of sequences which can be run at once.
    pub fn with_max_num_seqs(mut self, max_num_seqs: usize) -> Self {
        self.max_num_seqs = max_num_seqs;
        self
    }

    /// Disable KV cache. Trade performance for memory usage.
    pub fn with_no_kv_cache(mut self) -> Self {
        self.no_kv_cache = true;
        self
    }

    /// Set the number of sequences to hold in the prefix cache. Set to `None` to disable the prefix cacher.
    pub fn with_prefix_cache_n(mut self, n_seqs: Option<usize>) -> Self {
        self.prefix_cache_n = n_seqs;
        self
    }

    /// Enable logging.
    pub fn with_logging(mut self) -> Self {
        self.with_logging = true;
        self
    }

    /// Provide metadata to initialize the device mapper.
    pub fn with_device_mapping(mut self, device_mapping: DeviceMapSetting) -> Self {
        self.device_mapping = Some(device_mapping);
        self
    }

    pub async fn build(self) -> anyhow::Result<Model> {
        if self.with_logging {
            initialize_logging();
        }
        let path = Path::new(&self.model_id);
        let dtype = ModelDType::Auto;
        let device = best_device(self.force_cpu)?;
        let mapper = self
            .device_mapping
            .unwrap_or(DeviceMapSetting::Auto(AutoDeviceMapParams::default_text()));
        let loader = if path.is_file() {
            // Load from a GGUF
            let Some(model_filename) = path.file_name() else {
                bail!("Missing filename in model path");
            };
            let Some(model_dir) = path.parent() else {
                bail!("Invalid model path");
            };
            GGUFLoaderBuilder::new(
                None,
                None,
                model_dir.display().to_string(),
                vec![model_filename.to_string_lossy().into_owned()],
                GGUFSpecificConfig {
                    prompt_chunksize: None,
                    topology: None,
                },
            )
            .build()
        } else {
            // Load from a HF repo dir
            NormalLoaderBuilder::new(
                NormalSpecificConfig {
                    use_flash_attn: false,
                    prompt_chunksize: None,
                    topology: None,
                    organization: Default::default(),
                    write_uqff: None,
                    from_uqff: None,
                    imatrix: None,
                    calibration_file: None,
                },
                None,
                None,
                Some(self.model_id),
            )
            .build(None)?
        };
        // Load, into a Pipeline
        let pipeline = loader.load_model_from_hf(
            self.hf_revision,
            // The model was already downloaded
            TokenSource::None,
            &dtype,
            &device,
            !self.with_logging,
            mapper,
            None,
            self.paged_attn_cfg,
        )?;
        let scheduler_method = match self.paged_attn_cfg {
            Some(_) => {
                let config = pipeline
                    .lock()
                    .await
                    .get_metadata()
                    .cache_config
                    .as_ref()
                    .unwrap()
                    .clone();

                SchedulerConfig::PagedAttentionMeta {
                    max_num_seqs: self.max_num_seqs,
                    config,
                }
            }
            None => SchedulerConfig::DefaultScheduler {
                method: DefaultSchedulerMethod::Fixed(self.max_num_seqs.try_into()?),
            },
        };

        let mut runner = MistralRsBuilder::new(pipeline, scheduler_method)
            .with_no_kv_cache(self.no_kv_cache)
            .with_no_prefix_cache(self.prefix_cache_n.is_none());

        if let Some(n) = self.prefix_cache_n {
            runner = runner.with_prefix_cache_n(n)
        }

        Ok(Model::new(runner.build()))
    }
}
