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

    fn validate_input(&self, input: &str) -> Result<(), ToolError> {
        if input.trim().is_empty() {
            Err(ToolError::InvalidInput)
        } else {
            Ok(())
        }
    }
}

#[derive(Debug, thiserror::Error)]
#[error("Tool error")]
pub enum ToolError {
    #[error("Invalid input provided to the tool")]
    InvalidInput,
    #[error("The tool produced invalid output")]
    InvalidOutput,
    #[error("The tool is not available or not configured properly")]
    InvalidTool,
    #[error("An unknown error occurred: {0}")]
    Unknown(String),
}

pub struct DummyTool;

impl Tool for DummyTool {
    fn name(&self) -> &str {
        "dummy-tool"
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn description(&self) -> &str {
        "A dummy tool for demonstrating the Tool trait"
    }

    fn author(&self) -> &str {
        "Dummy Author"
    }

    fn run(&self, input: &str) -> Result<String, ToolError> {
        self.validate_input(input)?;
        Ok(format!("Processed input: {}", input))
    }
}
