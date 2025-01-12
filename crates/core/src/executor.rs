use crate::chat::{Completion, Request, ResponseContent, ResponseToolCalls, ToolCall};
use crate::knowledge::Knowledge;
use crate::tool::Tool;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Manages the execution of tasks using an LLM, tools, and (optionally) memory components.
pub struct Executor<M: Completion> {
    model: Arc<RwLock<M>>,
    knowledges: Arc<Vec<Box<dyn Knowledge>>>,
    tools: Arc<Vec<Box<dyn Tool>>>,
}

impl<M: Completion> Executor<M> {
    /// Creates a new `Executor` instance.
    pub fn new(
        model: Arc<RwLock<M>>,
        knowledges: Arc<Vec<Box<dyn Knowledge>>>,
        tools: Arc<Vec<Box<dyn Tool>>>,
    ) -> Self {
        Self {
            model,
            knowledges,
            tools,
        }
    }

    /// Executes the task by managing interactions between the LLM and tools.
    pub async fn invoke(&mut self, mut request: Request) -> Result<M::Response, String> {
        request.knowledges = {
            let mut enriched_knowledges = Vec::new();
            for knowledge in self.knowledges.iter() {
                let enriched = knowledge.enrich(&request.prompt);
                enriched_knowledges.push(enriched);
            }
            enriched_knowledges
        };

        // Interact with the LLM to get a response.
        let mut model = self.model.write().await;
        let response = model
            .completion(request.clone())
            .await
            .map_err(|e| format!("Model error: {}", e))?;

        let response_str = response.content();
        if response_str.trim().is_empty() {
            return Err("Received an empty response from the LLM.".to_string());
        }

        // Attempt to parse and execute a tool action.
        for call in response.toolcalls() {
            self.execute_tool(call).await?;
        }

        Ok(response)
    }

    /// Executes a tool action and returns the result.
    async fn execute_tool(&self, call: ToolCall) -> Result<String, String> {
        if let Some(tool) = self
            .tools
            .iter()
            .find(|t| t.name().eq_ignore_ascii_case(&call.function.name))
        {
            tool.run(&call.function.arguments)
                .await
                .map_err(|e| e.to_string())
        } else {
            Err(format!("Tool not found: {}", call.function.name))
        }
    }
}
