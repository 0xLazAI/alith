use alith::{Agent, Tool, LLM};
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
    ) -> Self {
        DelegateAgent {
            model,
            name,
            api_key,
            base_url,
            preamble,
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
        let mut agent = Agent::new(
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
        let result = rt.block_on(async { agent.prompt(&prompt).await });
        result.map_err(|e| napi::bindgen_prelude::Error::from_reason(e.to_string()))
    }

    #[napi]
    pub fn prompt(&self, prompt: String) -> Result<String> {
        self.prompt_with_tools(prompt, vec![])
    }
}
