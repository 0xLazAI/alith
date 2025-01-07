use crate::chat::Completion;
use crate::tool::Tool;
use regex::Regex;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Represents an action to be performed by a tool.
pub struct ToolAction {
    pub tool_name: String,
    pub input: String,
}

/// Manages the execution of tasks using an LLM, tools, and (optionally) memory components.
pub struct Executor<M: Completion> {
    model: Arc<RwLock<M>>,
    tools: Arc<Vec<Box<dyn Tool>>>,
    max_iterations: usize,
}

impl<M: Completion> Executor<M> {
    /// Creates a new `Executor` instance.
    pub fn new(
        model: Arc<RwLock<M>>,
        tools: Arc<Vec<Box<dyn Tool>>>,
        max_iterations: usize,
    ) -> Self {
        Self {
            model,
            tools,
            max_iterations,
        }
    }

    /// Executes the task by managing interactions between the LLM and tools.
    pub async fn invoke(&mut self, request: String) -> Result<M::Response, String> {
        let mut iterations = 0;

        while iterations < self.max_iterations {
            // Interact with the LLM to get a response.
            let mut model = self.model.write().await;
            let response = model
                .completion(request.clone().into())
                .await
                .map_err(|e| format!("Model error: {}", e))?;

            let response_str = response.to_string();
            if response_str.trim().is_empty() {
                return Err("Received an empty response from the LLM.".to_string());
            }

            // Check if the response contains the final answer.
            if self.is_final_answer(&response_str) {
                return Ok(response);
            }

            // Attempt to parse and execute a tool action.
            if let Some(action) = self.parse_action(&response_str) {
                self.execute_tool(&action).await?;
            }

            iterations += 1;
        }

        Err("Maximum iterations reached without a final answer.".to_string())
    }

    /// Determines whether the response contains the final answer.
    fn is_final_answer(&self, response: &str) -> bool {
        response.trim().starts_with("Final Answer:")
    }

    /// Parses a tool action from the LLM's response.
    fn parse_action(&self, response: &str) -> Option<ToolAction> {
        let re = Regex::new(r"Using tool: (\w+)\nInput: (.+)").ok()?;
        let captures = re.captures(response)?;

        Some(ToolAction {
            tool_name: captures.get(1)?.as_str().to_string(),
            input: captures.get(2)?.as_str().to_string(),
        })
    }

    /// Executes a tool action and returns the result.
    async fn execute_tool(&self, action: &ToolAction) -> Result<String, String> {
        if let Some(tool) = self
            .tools
            .iter()
            .find(|t| t.name().eq_ignore_ascii_case(&action.tool_name))
        {
            tool.run(&action.input).await.map_err(|e| e.to_string())
        } else {
            Err(format!("Tool not found: {}", action.tool_name))
        }
    }
}
