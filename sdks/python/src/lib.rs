use alith::{Agent, Tool, LLM};
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
    ) -> Self {
        DelegateAgent {
            model,
            name,
            api_key,
            base_url,
            preamble,
            tools,
        }
    }

    pub fn prompt(&self, prompt: &str) -> PyResult<String> {
        let tools = self
            .tools
            .iter()
            .map(|t| Box::new(t.clone()) as Box<dyn Tool>)
            .collect::<Vec<_>>();
        let mut agent = Agent::new(
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
        let result = rt.block_on(async { agent.prompt(prompt).await });
        result.map_err(|e| PyErr::new::<PyException, _>(e.to_string()))
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn _alith(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DelegateAgent>()?;
    m.add_class::<DelegateTool>()?;
    Ok(())
}
