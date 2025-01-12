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
    pub tools: Vec<DelegateTool>,
}

#[pymethods]
impl DelegateAgent {
    #[new]
    pub fn new(name: String, model: String, tools: Vec<DelegateTool>) -> Self {
        DelegateAgent { model, name, tools }
    }

    pub fn prompt(&self, prompt: &str) -> PyResult<String> {
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
