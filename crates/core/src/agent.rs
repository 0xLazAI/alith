use crate::chat::{Completion, Request, ResponseContent};
use crate::executor::Executor;
use crate::knowledge::Knowledge;
use crate::task::TaskError;
use crate::tool::Tool;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

pub struct Agent<M: Completion> {
    /// The model to use.
    pub model: Arc<RwLock<M>>,
    /// The tools to use.
    pub tools: Arc<Vec<Box<dyn Tool>>>,
    /// Knowledge sources for the agent.
    pub knowledges: Vec<Box<dyn Knowledge>>,
    /// The id of the agent.
    pub id: Uuid,
    /// The name of the agent.
    pub name: String,
    /// System prompt
    pub preamble: String,
    /// System format for the agent.
    pub system_template: String,
    /// Prompt format for the agent.
    pub prompt_template: String,
    /// Response format for the agent.
    pub response_template: String,
    /// Enable or disable the verbose mode.
    pub verbose: bool,
    /// The maximum requests per minute for the completion text.
    pub max_rpm: Option<usize>,
    /// Temperature of the model
    pub temperature: Option<f32>,
    /// Maximum number of tokens for the completion.
    pub max_tokens: Option<usize>,
    /// Maximum execution time for an agent to execute a task.
    pub max_execution_time: Option<usize>,
    /// Whether to respect the context window.
    pub respect_context_window: bool,
    /// Whether code execution is allowed.
    pub allow_code_execution: bool,
}

impl<M: Completion> Agent<M>
where
    M: Completion,
{
    pub fn new(
        model: Arc<RwLock<M>>,
        tools: Arc<Vec<Box<dyn Tool>>>,
        id: Uuid,
        name: String,
    ) -> Agent<M> {
        Agent {
            model,
            tools,
            id,
            name,
            preamble: String::new(),
            system_template: String::new(),
            prompt_template: String::new(),
            response_template: String::new(),
            verbose: false,
            max_rpm: None,
            temperature: None,
            max_tokens: None,
            max_execution_time: None,
            knowledges: Vec::new(),
            respect_context_window: false,
            allow_code_execution: false,
        }
    }

    pub async fn prompt(&mut self, prompt: &str) -> Result<String, TaskError> {
        let mut executor = Executor::new(self.model.clone(), self.tools.clone());
        let mut req: Request = Request::new(prompt.to_string(), self.preamble.clone());
        req.max_tokens = self.max_tokens;
        req.temperature = self.temperature;
        req.tools = self
            .tools
            .iter()
            .map(|tool| tool.definition())
            .collect::<Vec<_>>();
        let response = executor
            .invoke(req)
            .await
            .map_err(|_| TaskError::ExecutionError)?;

        Ok(response.content())
    }
}
