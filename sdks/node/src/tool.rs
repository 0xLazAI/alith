use std::ffi::{c_char, CString};

use alith::{Tool, ToolDefinition, ToolError};
use async_trait::async_trait;

use napi::JsNumber;
use napi_derive::napi;

#[napi(object)]
#[derive(Clone)]
pub struct DelegateTool {
    pub name: String,
    pub version: String,
    pub description: String,
    pub parameters: String,
    pub author: String,
    pub func_agent: JsNumber,
}

unsafe impl Send for DelegateTool {}
unsafe impl Sync for DelegateTool {}

impl DelegateTool {
    fn run_with_func_agent(
        &self,
        input: &str,
        func_agent: i64,
    ) -> std::result::Result<String, ToolError> {
        unsafe {
            let func_method: extern "C" fn(args: *const c_char) -> *const c_char =
                std::mem::transmute(func_agent);
            let c_input = CString::new(input).map_err(|_| ToolError::InvalidInput)?;
            let c_result = func_method(c_input.as_ptr());
            if c_result.is_null() {
                return Err(ToolError::InvalidOutput);
            }
            let result = {
                let c_str = std::ffi::CStr::from_ptr(c_result);
                c_str.to_string_lossy().into_owned()
            };
            Ok(result)
        }
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
        let func_agent = self
            .func_agent
            .get_int64()
            .map_err(|_| ToolError::InvalidInput)?;
        self.run_with_func_agent(input, func_agent)
    }
}
