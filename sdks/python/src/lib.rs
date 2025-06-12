use alith::{Agent, Chat, ClientConfig, LLM, TaskError, Tool};
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use std::collections::HashMap;

mod tool;

use tokio::runtime::Runtime;
use tool::DelegateTool;

#[pyclass]
pub struct DelegateAgent {
    agent: Agent<LLM>,
    mcp_config_path: String,
}

#[pymethods]
impl DelegateAgent {
    #[new]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: String,
        model: String,
        api_key: String,
        base_url: String,
        preamble: String,
        tools: Vec<DelegateTool>,
        extra_headers: HashMap<String, String>,
        mcp_config_path: String,
    ) -> PyResult<Self> {
        let tools = tools
            .iter()
            .map(|t| Box::new(t.clone()) as Box<dyn Tool>)
            .collect::<Vec<_>>();
        let config = ClientConfig::builder().extra_headers(extra_headers).build();
        let agent = Agent::new_with_tools(
            name,
            if base_url.is_empty() {
                LLM::from_model_name_and_config(&model, config)
                    .map_err(|e| PyErr::new::<PyException, _>(e.to_string()))?
            } else {
                LLM::openai_compatible_model_with_config(&api_key, &base_url, &model, config)
                    .map_err(|e| PyErr::new::<PyException, _>(e.to_string()))?
            },
            tools,
        )
        .preamble(preamble);
        let runtime = pyo3_async_runtimes::tokio::get_runtime();
        let agent = runtime
            .block_on(async {
                if !mcp_config_path.is_empty() {
                    agent.mcp_config_path(&mcp_config_path).await
                } else {
                    Ok(agent)
                }
            })
            .map_err(|e| PyErr::new::<PyException, _>(e.to_string()))?;
        Ok(DelegateAgent {
            agent,
            mcp_config_path,
        })
    }

    pub fn prompt(&mut self, prompt: &str) -> PyResult<String> {
        let rt = Runtime::new().map_err(|e| PyErr::new::<PyException, _>(e.to_string()))?;
        let result = rt.block_on(async {
            if !self.mcp_config_path.is_empty() {
                self.agent
                    .start_mcp_servers(&self.mcp_config_path)
                    .await
                    .map_err(TaskError::MCPError)?;
            }
            self.agent.prompt(prompt).await
        });
        result.map_err(|e| PyErr::new::<PyException, _>(e.to_string()))
    }

    pub fn chat(&mut self, prompt: &str, history: Vec<Message>) -> PyResult<String> {
        let rt = Runtime::new().map_err(|e| PyErr::new::<PyException, _>(e.to_string()))?;
        let result = rt.block_on(async {
            if !self.mcp_config_path.is_empty() {
                self.agent
                    .start_mcp_servers(&self.mcp_config_path)
                    .await
                    .map_err(TaskError::MCPError)?;
            }
            self.agent
                .chat(prompt, unsafe {
                    std::mem::transmute::<Vec<Message>, Vec<alith::core::chat::Message>>(history)
                })
                .await
        });
        result.map_err(|e| PyErr::new::<PyException, _>(e.to_string()))
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct Message {
    /// "system", "user", "assistant" or "tool".
    #[pyo3(get, set)]
    pub role: String,
    #[pyo3(get, set)]
    pub content: String,
}

#[pymethods]
impl Message {
    #[new]
    pub fn new(role: String, content: String) -> Self {
        Self { role, content }
    }
}

/// Runs the text chunker on the incoming text and returns the chunks as a vector of strings.
///
/// * `text` - The natural language text to chunk.
/// * `max_chunk_token_size` - The maxium token sized to be chunked to. Inclusive.
/// * `overlap_percent` - The percentage of overlap between chunks. Default is None.
#[pyfunction]
fn chunk_text(
    text: &str,
    max_chunk_token_size: u32,
    overlap_percent: f32,
) -> PyResult<Vec<String>> {
    Ok(alith::chunk_text(
        text,
        max_chunk_token_size,
        if overlap_percent == 0.0 {
            Some(overlap_percent)
        } else {
            None
        },
    )
    .map_err(|e| PyErr::new::<PyException, _>(e.to_string()))?
    .unwrap_or_default())
}

/// A Python module implemented in Rust.
#[pymodule]
fn _alith(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DelegateAgent>()?;
    m.add_class::<DelegateTool>()?;
    m.add_class::<Message>()?;
    m.add_function(wrap_pyfunction!(chunk_text, m)?)?;
    Ok(())
}
