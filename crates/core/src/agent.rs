use uuid::Uuid;

use crate::chat::Completion;
use crate::knowledge::Knowledge;
use crate::task::{Task, TaskError};
use crate::tool::Tool;

pub struct Agent<M: Completion> {
    /// The model to use.
    pub model: M,
    /// The tools to use.
    pub tools: Vec<Box<dyn Tool>>,
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
    M: Completion<Request = String, Response = String>,
{
    pub fn new(model: M, tools: Vec<Box<dyn Tool>>, id: Uuid, name: String) -> Agent<M> {
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

    pub async fn execute_task(
        &mut self,
        mut task: Task<M>,
        context: Option<&str>,
        tools: Option<Vec<Box<dyn Tool>>>,
    ) -> Result<String, TaskError> {
        let mut task_prompt = task.description.clone();

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

        // Apply additional tools if provided.
        if let Some(execution_tools) = tools {
            for tool in execution_tools {
                task_prompt.push_str(
                    &tool
                        .run(&task_prompt)
                        .unwrap_or_else(|_| format!("\nTool `{}` execution failed.", tool.name())),
                );
            }
        }

        // Generate a response using the model.
        let response = self
            .model
            .completion(task_prompt)
            .await
            .map_err(|_| TaskError::ExecutionError)?;

        // Update the task's output and return the result.
        task.output = Some(response.clone());
        Ok(response)
    }

    fn validate_docker_installation(&self) {
        if self.verbose {
            println!("Validating Docker installation...");
        }
        // Placeholder: Add actual logic for Docker validation.
    }
}
