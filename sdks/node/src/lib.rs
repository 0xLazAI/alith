use std::collections::HashMap;

use alith::{Agent, Chat, ClientConfig, LLM, Tool};
use napi::bindgen_prelude::*;
use napi_derive::napi;

mod tool;

use once_cell::sync::Lazy;
use tokio::runtime::Runtime;
use tool::DelegateTool;

pub(crate) static GLOBAL_RUNTIME: Lazy<Runtime> = Lazy::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
});

#[napi]
pub struct DelegateAgent {
    pub model: String,
    pub name: String,
    pub api_key: String,
    pub base_url: String,
    pub preamble: String,
    pub mcp_config_path: String,
    pub extra_headers: HashMap<String, String>,
}

/// Runs the text chunker on the incoming text and returns the chunks as a vector of strings.
///
/// * `text` - The natural language text to chunk.
/// * `max_chunk_token_size` - The maxium token sized to be chunked to. Inclusive.
/// * `overlap_percent` - The percentage of overlap between chunks. Default is None.
#[napi]
pub fn chunk_text(
    text: String,
    max_chunk_token_size: Option<u32>,
    overlap_percent: Option<f64>,
) -> Result<Vec<String>> {
    Ok(alith::chunk_text(
        &text,
        max_chunk_token_size.unwrap_or(200),
        overlap_percent.map(|p| p as f32),
    )
    .map_err(|e| napi::bindgen_prelude::Error::from_reason(e.to_string()))?
    .unwrap_or_default())
}

#[napi]
impl DelegateAgent {
    #[napi(constructor)]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: String,
        model: String,
        api_key: String,
        base_url: String,
        preamble: String,
        mcp_config_path: String,
        extra_headers: HashMap<String, String>,
    ) -> Self {
        DelegateAgent {
            model,
            name,
            api_key,
            base_url,
            preamble,
            mcp_config_path,
            extra_headers,
        }
    }

    #[napi]
    pub fn prompt_with_tools(
        &self,
        prompt: String,
        history: Vec<Message>,
        delegate_tools: Vec<DelegateTool>,
    ) -> Result<String> {
        let mut tools = vec![];
        for tool in delegate_tools {
            tools.push(Box::new(tool) as Box<dyn Tool>);
        }
        let config = ClientConfig::builder()
            .extra_headers(self.extra_headers.clone())
            .build();
        let mut agent = Agent::new_with_tools(
            self.name.to_string(),
            if self.base_url.is_empty() {
                LLM::from_model_name_and_config(&self.model, config)
                    .map_err(|e| napi::bindgen_prelude::Error::from_reason(e.to_string()))?
            } else {
                LLM::openai_compatible_model_with_config(
                    &self.api_key,
                    &self.base_url,
                    &self.model,
                    config,
                )
                .map_err(|e| napi::bindgen_prelude::Error::from_reason(e.to_string()))?
            },
            tools,
        );
        let history = unsafe {
            std::mem::transmute::<Vec<Message>, Vec<alith::core::chat::Message>>(history)
        };
        agent.preamble = self.preamble.clone();
        let result = GLOBAL_RUNTIME.block_on(async {
            if !self.mcp_config_path.is_empty() {
                agent = agent.mcp_config_path(&self.mcp_config_path).await?;
            }
            agent.chat(&prompt, history).await
        });
        result.map_err(|e| napi::bindgen_prelude::Error::from_reason(e.to_string()))
    }

    #[napi]
    pub fn prompt(&self, prompt: String) -> Result<String> {
        self.prompt_with_tools(prompt, vec![], vec![])
    }

    #[napi]
    pub fn chat(&self, prompt: String, history: Vec<Message>) -> Result<String> {
        self.prompt_with_tools(prompt, history, vec![])
    }
}

#[napi(object)]
#[derive(Clone, Debug)]
pub struct Message {
    /// "system", "user", "assistant" or "tool".
    pub role: String,
    pub content: String,
}
