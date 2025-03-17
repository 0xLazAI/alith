use alith::{Agent, LLM, Tool};
use pyo3::exceptions::PyException;
use pyo3::prelude::*;

mod tool;

use tokio::runtime::Runtime;
use tool::DelegateTool;

#[pyclass]
#[derive(Clone)]
pub struct DelegateAgent {
    #[pyo3(get, set)]
    pub model: String,
    #[pyo3(get, set)]
    pub name: String,
    #[pyo3(get, set)]
    pub api_key: String,
    #[pyo3(get, set)]
    pub base_url: String,
    #[pyo3(get, set)]
    pub preamble: String,
    #[pyo3(get, set)]
    pub tools: Vec<DelegateTool>,
    #[pyo3(get, set)]
    pub mcp_config_path: String,
}

#[pymethods]
impl DelegateAgent {
    #[new]
    pub fn new(
        name: String,
        model: String,
        api_key: String,
        base_url: String,
        preamble: String,
        tools: Vec<DelegateTool>,
        mcp_config_path: String,
    ) -> Self {
        DelegateAgent {
            model,
            name,
            api_key,
            base_url,
            preamble,
            tools,
            mcp_config_path,
        }
    }

    pub fn prompt(&self, prompt: &str) -> PyResult<String> {
        let tools = self
            .tools
            .iter()
            .map(|t| Box::new(t.clone()) as Box<dyn Tool>)
            .collect::<Vec<_>>();
        let mut agent = Agent::new_with_tools(
            self.name.to_string(),
            if self.base_url.is_empty() {
                LLM::from_model_name(&self.model)
                    .map_err(|e| PyErr::new::<PyException, _>(e.to_string()))?
            } else {
                LLM::openai_compatible_model(&self.api_key, &self.base_url, &self.model)
                    .map_err(|e| PyErr::new::<PyException, _>(e.to_string()))?
            },
            tools,
        );
        agent.preamble = self.preamble.clone();
        let rt = Runtime::new().map_err(|e| PyErr::new::<PyException, _>(e.to_string()))?;
        let result = rt.block_on(async {
            if !self.mcp_config_path.is_empty() {
                agent = agent.mcp_config_path(&self.mcp_config_path).await?;
            }
            agent.prompt(prompt).await
        });
        result.map_err(|e| PyErr::new::<PyException, _>(e.to_string()))
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
    m.add_function(wrap_pyfunction!(chunk_text, m)?)?;
    Ok(())
}
