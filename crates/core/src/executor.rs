use crate::Ref;
use crate::chat::{Completion, Request, ResponseContent, ResponseToolCalls, ToolCall};
use crate::knowledge::Knowledge;
use crate::mcp::MCPClient;
use crate::memory::{Memory, Message};
use crate::tool::Tool;
use std::sync::Arc;

/// Manages the execution of tasks using an LLM, tools, and (optionally) memory components.
pub struct Executor<M: Completion> {
    model: Ref<M>,
    knowledges: Arc<Vec<Box<dyn Knowledge>>>,
    tools: Ref<Vec<Box<dyn Tool>>>,
    memory: Option<Ref<dyn Memory>>,
    /// The MCP client used to communicate with the MCP server
    mcp_clients: Ref<Vec<MCPClient>>,
}

impl<M: Completion> Executor<M> {
    /// Creates a new `Executor` instance.
    pub fn new(
        model: Ref<M>,
        knowledges: Arc<Vec<Box<dyn Knowledge>>>,
        tools: Ref<Vec<Box<dyn Tool>>>,
        memory: Option<Ref<dyn Memory>>,
        mcp_clients: Ref<Vec<MCPClient>>,
    ) -> Self {
        Self {
            model,
            knowledges,
            tools,
            memory,
            mcp_clients,
        }
    }

    /// Executes the task by managing interactions between the LLM and tools.
    pub async fn invoke(&mut self, mut request: Request) -> anyhow::Result<String> {
        request.knowledges = {
            let mut enriched_knowledges = Vec::new();
            for knowledge in self.knowledges.iter() {
                let enriched = knowledge.enrich(&request.prompt)?;
                enriched_knowledges.push(enriched);
            }
            enriched_knowledges
        };
        // Add user memory
        self.add_user_message(&request.prompt).await;
        // Interact with the LLM to get a response.
        let mut model = self.model.write().await;
        let response = model.completion(request.clone()).await?;

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
    async fn add_user_message(&self, message: &str) {
        if let Some(memory) = &self.memory {
            let mut memory = memory.write().await;
            memory.add_user_message(message);
        }
    }

    /// Add an AI message into the memory if the memory has been set.
    async fn add_ai_message(&self, message: &str) {
        if let Some(memory) = &self.memory {
            let mut memory = memory.write().await;
            memory.add_ai_message(message);
        }
    }

    /// Add an AI message into the memory if the memory has been set.
    async fn add_ai_message_with_tool_call(&self, tool_call: &str) -> anyhow::Result<()> {
        if let Some(memory) = &self.memory {
            let mut memory = memory.write().await;
            let tool_call: serde_json::Value = serde_json::from_str(tool_call)?;
            memory.add_message(Message::new_ai_message("").with_tool_calls(tool_call));
        }
        Ok(())
    }

    /// Executes a tool action and returns the result.
    async fn execute_tool(&self, call: ToolCall) -> anyhow::Result<String> {
        let tools = self.tools.read().await;
        if let Some(tool) = tools
            .iter()
            .find(|t| t.name().eq_ignore_ascii_case(&call.function.name))
        {
            Ok(tool.run(&call.function.arguments).await?)
        } else {
            let mcp_clients = self.mcp_clients.read().await;
            if !mcp_clients.is_empty() {
                for mcp_client in mcp_clients.iter() {
                    if mcp_client.tools.contains_key(&call.function.name) {
                        let arguments = serde_json::from_str(&call.function.arguments)?;
                        let response = mcp_client.call_tool(&call.function.name, arguments).await?;
                        if let Some(text) = response.content[0].as_text() {
                            return Ok(text.to_string());
                        } else {
                            return Ok("".to_string());
                        }
                    }
                }
            }
            Err(anyhow::anyhow!("Tool not found: {}", call.function.name))
        }
    }
}
