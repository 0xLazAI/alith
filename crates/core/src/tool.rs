use async_trait::async_trait;
use schemars::{schema::RootSchema, schema_for, JsonSchema};
use serde::{Deserialize, Serialize};
use serde_json::json;

pub use llm_client::interface::requests::completion::{ToolChoice, ToolDefinition};

#[async_trait]
pub trait Tool: Send + Sync {
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

    fn definition(&self) -> ToolDefinition;

    fn validate_input(&self, input: &str) -> Result<(), ToolError> {
        if input.trim().is_empty() {
            Err(ToolError::InvalidInput)
        } else {
            Ok(())
        }
    }

    async fn run(&self, input: &str) -> Result<String, ToolError>;
}

#[async_trait]
pub trait StructureTool: Send + Sync {
    type Input: for<'a> Deserialize<'a> + JsonSchema + Send + Sync;
    type Output: Serialize;

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

    fn schema(&self) -> RootSchema {
        schema_for!(Self::Input)
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.name().to_owned(),
            description: self.description().to_owned(),
            parameters: json!(self.schema()),
        }
    }

    async fn run_with_args(&self, input: Self::Input) -> Result<Self::Output, ToolError>;

    async fn run(&self, input: &str) -> Result<String, ToolError> {
        match serde_json::from_str(input) {
            Ok(input) => {
                let output = self.run_with_args(input).await?;
                serde_json::to_string(&output).map_err(ToolError::JsonError)
            }
            Err(e) => Err(ToolError::JsonError(e)),
        }
    }
}

#[async_trait]
impl<T: StructureTool> Tool for T {
    fn name(&self) -> &str {
        self.name()
    }

    fn version(&self) -> &str {
        self.version()
    }

    fn description(&self) -> &str {
        self.description()
    }

    fn author(&self) -> &str {
        self.author()
    }

    fn definition(&self) -> ToolDefinition {
        self.definition()
    }

    async fn run(&self, input: &str) -> Result<String, ToolError> {
        match serde_json::from_str(input) {
            Ok(input) => {
                let output = self.run_with_args(input).await?;
                serde_json::to_string(&output).map_err(ToolError::JsonError)
            }
            Err(e) => Err(ToolError::JsonError(e)),
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
    #[error("JsonError: {0}")]
    JsonError(#[from] serde_json::Error),
}

#[cfg(test)]
mod tests {
    use super::{StructureTool, Tool, ToolError};
    use async_trait::async_trait;
    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};
    use serde_json::json;

    pub struct DummyTool;

    #[derive(JsonSchema, Serialize, Deserialize)]
    pub struct DummpyInput {
        pub x: usize,
        pub y: usize,
    }

    #[async_trait]
    impl StructureTool for DummyTool {
        type Input = DummpyInput;
        type Output = String;

        fn name(&self) -> &str {
            "dummy"
        }

        async fn run_with_args(&self, input: Self::Input) -> Result<Self::Output, ToolError> {
            Ok(format!("x: {}, y: {}", input.x, input.y))
        }
    }

    #[tokio::test]
    async fn test_dummy_tool() {
        let tool: Box<dyn Tool> = Box::new(DummyTool);
        let output = tool
            .run(
                serde_json::to_string(&json!({
                    "x": 1,
                    "y": 2
                }))
                .unwrap()
                .as_str(),
            )
            .await
            .unwrap();
        assert_eq!(tool.name(), "dummy");
        assert_eq!(output, "\"x: 1, y: 2\"");
        assert_eq!(
            tool.definition().parameters.to_string(),
            "{\"$schema\":\"http://json-schema.org/draft-07/schema#\",\"properties\":{\"x\":{\"format\":\"uint\",\"minimum\":0.0,\"type\":\"integer\"},\"y\":{\"format\":\"uint\",\"minimum\":0.0,\"type\":\"integer\"}},\"required\":[\"x\",\"y\"],\"title\":\"DummpyInput\",\"type\":\"object\"}"
        );
    }
}
