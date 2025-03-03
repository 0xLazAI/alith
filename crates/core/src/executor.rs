use crate::chat::{Completion, Request, ResponseContent, ResponseToolCalls, ToolCall};
use crate::knowledge::Knowledge;
use crate::memory::{Memory, Message};
use crate::tool::Tool;
use crate::Ref;
use anyhow::Result;
use mcp_client::McpClientTrait;
use std::sync::Arc;

/// Manages the execution of tasks using an LLM, tools, and (optionally) memory components.
pub struct Executor<M: Completion> {
    model: Ref<M>,
    knowledges: Arc<Vec<Box<dyn Knowledge>>>,
    tools: Ref<Vec<Box<dyn Tool>>>,
    memory: Option<Ref<dyn Memory>>,
    /// The MCP client used to communicate with the MCP server
    mcp_client: Option<Arc<dyn McpClientTrait>>,
}

impl<M: Completion> Executor<M> {
    /// Creates a new `Executor` instance.
    pub fn new(
        model: Ref<M>,
        knowledges: Arc<Vec<Box<dyn Knowledge>>>,
        tools: Ref<Vec<Box<dyn Tool>>>,
        memory: Option<Ref<dyn Memory>>,
        mcp_client: Option<Arc<dyn McpClientTrait>>,
    ) -> Self {
        Self {
            model,
            knowledges,
            tools,
            memory,
            mcp_client,
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

        let mut responses = vec![response.content()];
        self.add_ai_message(&responses[0]).await;

        // Attempt to parse and execute a tool action.
        for call in response.toolcalls() {
            let tool_call = self.execute_tool(call).await?;
            self.add_ai_message_with_tool_call(&tool_call).await?;
            responses.push(tool_call);
        }

        Ok(responses.join("\n"))
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
        let tools = self.tools.read().await;
        if let Some(tool) = tools
            .iter()
            .find(|t| t.name().eq_ignore_ascii_case(&call.function.name))
        {
            tool.run(&call.function.arguments)
                .await
                .map_err(|e| e.to_string())
        } else if let Some(mcp_client) = &self.mcp_client {
            let arguments =
                serde_json::from_str(&call.function.arguments).map_err(|e| e.to_string())?;
            let response = mcp_client
                .call_tool(&call.function.name, arguments)
                .await
                .map_err(|e| format!("MCP error: {}", e))?;
            if let Some(text) = response.content[0].as_text() {
                Ok(text.to_string())
            } else {
                Ok("".to_string())
            }
        } else {
            Err(format!("Tool not found: {}", call.function.name))
        }
    }
}
