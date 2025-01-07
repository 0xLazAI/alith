use std::ffi::{c_char, CString};

use alith::{Tool, ToolDefinition, ToolError};
use async_trait::async_trait;
use pyo3::prelude::*;

#[pyclass]
#[derive(Clone)]
pub struct DelegateTool {
    #[pyo3(get, set)]
    pub name: String,
    #[pyo3(get, set)]
    pub version: String,
    #[pyo3(get, set)]
    pub description: String,
    #[pyo3(get, set)]
    pub parameters: String,
    #[pyo3(get, set)]
    pub author: String,
    #[pyo3(get, set)]
    pub func_agent: u64,
}

#[pymethods]
impl DelegateTool {
    #[new]
    pub fn new(
        name: String,
        version: String,
        description: String,
        parameters: String,
        author: String,
        func_agent: u64,
    ) -> Self {
        DelegateTool {
            name,
            version,
            description,
            parameters,
            author,
            func_agent,
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

    async fn run(&self, input: &str) -> Result<String, ToolError> {
        unsafe {
            let func_method: extern "C" fn(args: *const c_char) -> *const c_char =
                std::mem::transmute(self.func_agent);
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
