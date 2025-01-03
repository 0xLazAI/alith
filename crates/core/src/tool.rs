pub trait Tool {
    fn name(&self) -> &str {
        "default-tool"
    }
    fn version(&self) -> &str {
        "0.0.0"
    }
    fn description(&self) -> &str {
        "A default tool"
    }
    fn author(&self) -> &str {
        "Anonymous"
    }
    fn run(&self, input: &str) -> Result<String, ToolError>;
}

#[derive(Debug, thiserror::Error)]
#[error("Tool error")]
pub enum ToolError {
    InvalidInput,
    InvalidOutput,
    InvalidTool,
}
