use crate::chat::Completion;
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
    /// The id of the agent.
    pub id: Uuid,
    /// The name of the agent.
    pub name: String,
    /// Role of the agent.
    pub role: String,
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
    /// Maximum number of tokens for the completion.
    pub max_tokens: Option<usize>,
    /// Maximum execution time for an agent to execute a task.
    pub max_execution_time: Option<usize>,
    /// Knowledge sources for the agent.
    pub knowledges: Vec<Box<dyn Knowledge>>,
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
            role: "default".to_string(),
            preamble: String::new(),
            system_template: String::new(),
            prompt_template: String::new(),
            response_template: String::new(),
            verbose: false,
            max_rpm: None,
            max_tokens: None,
            max_execution_time: None,
            knowledges: Vec::new(),
            respect_context_window: false,
            allow_code_execution: false,
        }
    }

    pub async fn completion(
        &mut self,
        request: M::Request,
        context: Option<&str>,
    ) -> Result<M::Response, TaskError> {
        let mut task_prompt = request.to_string();

        // Add context if provided.
        if let Some(context) = context {
            task_prompt = format!("{}\nContext: {}", task_prompt, context);
        }

        // Apply context window constraints if enabled.
        if self.respect_context_window {
            // Placeholder for context window logic (e.g., truncation or summarization).
        }

        // Enrich the task prompt using knowledge sources.
        for knowledge in &self.knowledges {
            task_prompt = knowledge.enrich(&task_prompt);
        }

        // Validate Docker installation for code execution (if allowed).
        if self.allow_code_execution {
            self.validate_docker_installation();
        }

        let mut executor = Executor::new(self.model.clone(), self.tools.clone(), 10);
        let response = executor
            .invoke(task_prompt)
            .await
            .map_err(|_| TaskError::ExecutionError)?;

        Ok(response)
    }

    fn validate_docker_installation(&self) {
        if self.verbose {
            println!("Validating Docker installation...");
        }
        // Placeholder: Add actual logic for Docker validation.
    }
}
