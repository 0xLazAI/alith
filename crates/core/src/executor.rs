use crate::chat::{Completion, Request, ResponseContent, ResponseToolCalls, ToolCall};
use crate::knowledge::Knowledge;
use crate::memory::{Memory, Message};
use crate::tool::Tool;
use anyhow::Result;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Manages the execution of tasks using an LLM, tools, and (optionally) memory components.
pub struct Executor<M: Completion> {
    model: Arc<RwLock<M>>,
    knowledges: Arc<Vec<Box<dyn Knowledge>>>,
    tools: Arc<Vec<Box<dyn Tool>>>,
    memory: Option<Arc<RwLock<dyn Memory>>>,
}

impl<M: Completion> Executor<M> {
    /// Creates a new `Executor` instance.
    pub fn new(
        model: Arc<RwLock<M>>,
        knowledges: Arc<Vec<Box<dyn Knowledge>>>,
        tools: Arc<Vec<Box<dyn Tool>>>,
        memory: Option<Arc<RwLock<dyn Memory>>>,
    ) -> Self {
        Self {
            model,
            knowledges,
            tools,
            memory,
        }
    }

    /// Executes the task by managing interactions between the LLM and tools.
    pub async fn invoke(&mut self, mut request: Request) -> Result<String, String> {
        request.knowledges = {
            let mut enriched_knowledges = Vec::new();
            for knowledge in self.knowledges.iter() {
                let enriched = knowledge
                    .enrich(&request.prompt)
                    .map_err(|err| err.to_string())?;
                enriched_knowledges.push(enriched);
            }
            enriched_knowledges
        };
        // Add user memory
        self.add_user_message(&request.prompt).await;
        // Interact with the LLM to get a response.
        let mut model = self.model.write().await;
        let response = model
            .completion(request.clone())
            .await
            .map_err(|e| format!("Model error: {}", e))?;

        let mut response_str = response.content();
        self.add_ai_message(&response_str).await;

        // Attempt to parse and execute a tool action.
        for call in response.toolcalls() {
            let tool_call = self.execute_tool(call).await?;
            self.add_ai_message_with_tool_call(&tool_call).await?;
            response_str.push_str(&tool_call);
        }

        Ok(response_str)
    }

    /// Add a user message into the memory if the memory has been set.
    async fn add_user_message(&self, message: &dyn std::fmt::Display) {
        if let Some(memory) = &self.memory {
            let mut memory = memory.write().await;
            memory.add_user_message(message);
        }
    }

    /// Add an AI message into the memory if the memory has been set.
    async fn add_ai_message(&self, message: &dyn std::fmt::Display) {
        if let Some(memory) = &self.memory {
            let mut memory = memory.write().await;
            memory.add_ai_message(message);
        }
    }

    /// Add an AI message into the memory if the memory has been set.
    async fn add_ai_message_with_tool_call(
        &self,
        tool_call: &dyn std::fmt::Display,
    ) -> Result<(), String> {
        if let Some(memory) = &self.memory {
            let mut memory = memory.write().await;
            let tool_call: serde_json::Value =
                serde_json::from_str(&format!("{tool_call}")).map_err(|err| err.to_string())?;
            memory.add_message(Message::new_ai_message("").with_tool_calls(tool_call));
        }
        Ok(())
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
