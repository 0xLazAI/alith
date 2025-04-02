use std::marker::PhantomData;

use crate::{
    agent::Agent,
    chat::{Chat, Completion},
    task::TaskError,
    tool::{StructureTool, ToolError},
};
use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

pub struct Extractor<M>
where
    M: Completion,
{
    agent: Agent<M>,
}

impl<M> Extractor<M>
where
    M: Completion + Send + Sync,
{
    /// Constructor for Extractor that initializes the agent with the given model.
    #[inline]
    pub async fn new<T>(model: M) -> Self
    where
        T: Serialize + for<'a> Deserialize<'a> + JsonSchema + Send + Sync + 'static,
    {
        Self {
            agent: Agent::new("extract-agent", model)
                .preamble(
                    r#"Extract the data structure from the input string.
Note you MUST use the tool named `extractor` to extract the input string to the
data structure.
"#,
                )
                .tool(ExtractTool::<T> { _data: PhantomData })
                .await,
        }
    }

    /// Extract structure data from an input string.
    #[inline]
    pub async fn extract<T>(&self, input: &str) -> Result<T, ExtractionError>
    where
        T: Serialize + for<'a> Deserialize<'a> + JsonSchema + Send + Sync + 'static,
    {
        Ok(serde_json::from_str(&self.agent.prompt(input).await?)?)
    }
}

#[derive(Debug, thiserror::Error)]
#[error("Extraction error")]
pub enum ExtractionError {
    #[error("TaskError: {0}")]
    TaskError(#[from] TaskError),
    #[error("JsonError: {0}")]
    JsonError(#[from] serde_json::Error),
}

struct ExtractTool<T>
where
    T: Serialize + for<'a> Deserialize<'a> + JsonSchema + Send + Sync,
{
    _data: PhantomData<T>,
}

#[async_trait]
impl<T> StructureTool for ExtractTool<T>
where
    T: Serialize + for<'a> Deserialize<'a> + JsonSchema + Send + Sync,
{
    type Input = T;
    type Output = T;

    fn name(&self) -> &str {
        "extractor"
    }

    fn description(&self) -> &str {
        "Extract the data structure from the input string."
    }

    async fn run_with_args(&self, input: Self::Input) -> Result<Self::Output, ToolError> {
        Ok(input)
    }
}
