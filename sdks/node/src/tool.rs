use alith::{Tool, ToolDefinition, ToolError};
use async_trait::async_trait;

use napi::{Env, JsFunction, JsString};
use napi_derive::napi;

use crate::sys;

use super::ValueType;

#[derive(Clone, Copy)]
pub struct Value {
    pub env: sys::napi_env,
    _value: sys::napi_value,
    _value_type: ValueType,
}

#[napi(object)]
pub struct DelegateTool {
    pub name: String,
    pub version: String,
    pub description: String,
    pub parameters: String,
    pub author: String,
    pub handler: JsFunction,
}

unsafe impl Send for DelegateTool {}
unsafe impl Sync for DelegateTool {}

impl DelegateTool {
    fn run_with_func_agent(
        &self,
        input: &str,
        func_agent: &JsFunction,
    ) -> std::result::Result<String, ToolError> {
        let func_value: &Value = unsafe { std::mem::transmute(func_agent) };
        let env = unsafe { Env::from_raw(func_value.env) };
        let js_input = env
            .create_string(input)
            .map_err(|_| ToolError::InvalidInput)?;
        let result = self
            .handler
            .call(None, &[js_input])
            .map_err(|_| ToolError::InvalidOutput)?;
        let result_str: JsString = result
            .coerce_to_string()
            .map_err(|_| ToolError::InvalidOutput)?;
        Ok(result_str
            .into_utf8()
            .map_err(|_| ToolError::InvalidOutput)?
            .as_str()
            .map_err(|_| ToolError::InvalidOutput)?
            .to_string())
    }
}

#[async_trait]
impl Tool for DelegateTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn version(&self) -> &str {
        &self.version
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn author(&self) -> &str {
        &self.author
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.name.to_string(),
            description: self.description.to_string(),
            parameters: serde_json::from_str(&self.parameters).unwrap(),
        }
    }

    async fn run(&self, input: &str) -> std::result::Result<String, ToolError> {
        self.run_with_func_agent(input, &self.handler)
    }
}
