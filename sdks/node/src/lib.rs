use alith::{Agent, LLM, Tool};
use napi::bindgen_prelude::*;
use napi_derive::napi;

mod tool;

use tokio::runtime::Runtime;
use tool::DelegateTool;

#[napi]
pub struct DelegateAgent {
    pub model: String,
    pub name: String,
    pub api_key: String,
    pub base_url: String,
    pub preamble: String,
    pub mcp_config_path: String,
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
    pub fn new(
        name: String,
        model: String,
        api_key: String,
        base_url: String,
        preamble: String,
        mcp_config_path: String,
    ) -> Self {
        DelegateAgent {
            model,
            name,
            api_key,
            base_url,
            preamble,
            mcp_config_path,
        }
    }

    #[napi]
    pub fn prompt_with_tools(
        &self,
        prompt: String,
        delegate_tools: Vec<DelegateTool>,
    ) -> Result<String> {
        let mut tools = vec![];
        for tool in delegate_tools {
            tools.push(Box::new(tool) as Box<dyn Tool>);
        }
        let mut agent = Agent::new_with_tools(
            self.name.to_string(),
            if self.base_url.is_empty() {
                LLM::from_model_name(&self.model)
                    .map_err(|e| napi::bindgen_prelude::Error::from_reason(e.to_string()))?
            } else {
                LLM::openai_compatible_model(&self.api_key, &self.base_url, &self.model)
                    .map_err(|e| napi::bindgen_prelude::Error::from_reason(e.to_string()))?
            },
            tools,
        );
        agent.preamble = self.preamble.clone();
        let rt =
            Runtime::new().map_err(|e| napi::bindgen_prelude::Error::from_reason(e.to_string()))?;
        let result = rt.block_on(async {
            if !self.mcp_config_path.is_empty() {
                agent = agent.mcp_config_path(&self.mcp_config_path).await?;
            }
            agent.prompt(&prompt).await
        });
        result.map_err(|e| napi::bindgen_prelude::Error::from_reason(e.to_string()))
    }

    #[napi]
    pub fn prompt(&self, prompt: String) -> Result<String> {
        self.prompt_with_tools(prompt, vec![])
    }
}
