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
    pub tools: Vec<DelegateTool>,
}

#[napi]
impl DelegateAgent {
    #[napi(constructor)]
    pub fn new(name: String, model: String, tools: Vec<DelegateTool>) -> Self {
        DelegateAgent { model, name, tools }
    }

    #[napi]
    pub fn prompt(&self, prompt: String) -> Result<String> {
        let tools = self
            .tools
            .iter()
            .map(|t| Box::new(t.clone()) as Box<dyn Tool>)
            .collect::<Vec<_>>();
        let mut agent = Agent::new(
            self.name.to_string(),
            LLM::from_model_name(&self.model).unwrap(),
            tools,
        );
        let rt = Runtime::new().unwrap();
        let result = rt.block_on(async { agent.prompt(&prompt).await });
        result.map_err(|e| napi::bindgen_prelude::Error::from_reason(e.to_string()))
    }
}
