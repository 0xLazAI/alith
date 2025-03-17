use crate::chat::{Completion, Document, Message, Request};
use crate::executor::Executor;
use crate::knowledge::Knowledge;
use crate::mcp::{MCPClient, MCPError, setup_mcp_clients, sse_client, stdio_client};
use crate::memory::Memory;
use crate::store::{Storage, VectorStoreError};
use crate::task::TaskError;
use crate::tool::Tool;
use crate::{Ref, make_ref};
use futures::{StreamExt, TryStreamExt, stream};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

pub struct Agent<M: Completion> {
    /// The model to use.
    pub model: Ref<M>,
    /// Indexed storage for the agent.
    pub store_indices: Vec<(usize, Box<dyn Storage>)>,
    /// The tools to use.
    pub tools: Ref<Vec<Box<dyn Tool>>>,
    /// Knowledge sources for the agent.
    pub knowledges: Arc<Vec<Box<dyn Knowledge>>>,
    /// Agent memory.
    pub memory: Option<Ref<dyn Memory>>,
    /// The unique ID of the agent.
    pub id: Uuid,
    /// The name of the agent.
    pub name: String,
    /// System prompt for the agent.
    pub preamble: String,
    /// Temperature of the model.
    pub temperature: Option<f32>,
    /// Maximum number of tokens for the completion.
    pub max_tokens: Option<usize>,
    /// Maximum execution time for the agent to complete a task.
    pub max_execution_time: Option<usize>,
    /// Whether to respect the context window.
    pub respect_context_window: bool,
    /// Whether code execution is allowed.
    pub allow_code_execution: bool,
    /// The MCP client used to communicate with the MCP server
    mcp_clients: Ref<Vec<MCPClient>>,
}

impl<M: Completion> Agent<M>
where
    M: Completion,
{
    /// Creates a new agent.
    pub fn new(name: impl ToString, model: M) -> Agent<M> {
        Agent {
            model: Arc::new(RwLock::new(model)),
            tools: make_ref(vec![]),
            store_indices: vec![],
            id: Uuid::new_v4(),
            name: name.to_string(),
            preamble: String::new(),
            temperature: None,
            max_tokens: None,
            max_execution_time: None,
            knowledges: Arc::new(Vec::new()),
            memory: None,
            mcp_clients: make_ref(vec![]),
            respect_context_window: false,
            allow_code_execution: false,
        }
    }

    /// Creates a new agent with some tools
    pub fn new_with_tools<I>(name: impl ToString, model: M, tools: I) -> Agent<M>
    where
        I: IntoIterator<Item = Box<dyn Tool>>,
    {
        Agent {
            model: Arc::new(RwLock::new(model)),
            tools: make_ref(tools.into_iter().collect()),
            store_indices: vec![],
            id: Uuid::new_v4(),
            name: name.to_string(),
            preamble: String::new(),
            temperature: None,
            max_tokens: None,
            max_execution_time: None,
            knowledges: Arc::new(Vec::new()),
            memory: None,
            mcp_clients: make_ref(vec![]),
            respect_context_window: false,
            allow_code_execution: false,
        }
    }

    /// Add a tool into the agent
    pub async fn tool(self, tool: impl Tool + 'static) -> Self {
        let mut self_tools = self.tools.write().await;
        self_tools.push(Box::new(tool));
        drop(self_tools);
        self
    }

    /// Add some tools into the agent
    pub async fn tools<I>(self, tools: I) -> Self
    where
        I: IntoIterator<Item = Box<dyn Tool>>,
    {
        let mut self_tools = self.tools.write().await;
        for tool in tools.into_iter() {
            self_tools.push(tool);
        }
        drop(self_tools);
        self
    }

    /// Adds a memory to the agent.
    pub fn memory(mut self, memory: impl Memory + 'static) -> Self {
        self.memory = Some(Arc::new(RwLock::new(memory)));
        self
    }

    /// Adds a storage index to the agent.
    pub fn store_index(mut self, sample: usize, store: impl Storage + 'static) -> Self {
        self.store_indices.push((sample, Box::new(store)));
        self
    }

    /// System prompt for the agent.
    pub fn preamble(mut self, preamble: impl ToString) -> Self {
        self.preamble = preamble.to_string();
        self
    }

    /// Set the MCP client.
    pub async fn mcp_client(self, mcp_client: MCPClient) -> Self {
        let mut mcp_clients = self.mcp_clients.write().await;
        mcp_clients.push(mcp_client);
        drop(mcp_clients);
        self
    }

    /// Set the MCP server config path.
    pub async fn mcp_config_path<P: AsRef<Path>>(self, path: P) -> anyhow::Result<Self, MCPError> {
        let clients = setup_mcp_clients(path).await?;
        let mut mcp_clients = self.mcp_clients.write().await;
        for (_, client) in clients {
            mcp_clients.push(client);
        }
        drop(mcp_clients);
        Ok(self)
    }

    /// Set the MCP sse client.
    #[inline]
    pub async fn mcp_sse_client<S: AsRef<str> + 'static>(
        self,
        sse_url: S,
        env: HashMap<String, String>,
    ) -> anyhow::Result<Self> {
        Ok(self.mcp_client(sse_client(sse_url, env).await?).await)
    }

    /// Set the MCP sse client.
    #[inline]
    pub async fn mcp_stdio_client<S: AsRef<str> + 'static>(
        self,
        command: S,
        args: Vec<S>,
        env: HashMap<String, String>,
    ) -> anyhow::Result<Self> {
        Ok(self
            .mcp_client(stdio_client(command, args, env).await?)
            .await)
    }

    /// Processes a prompt using the agent.
    pub async fn prompt(&self, prompt: &str) -> Result<String, TaskError> {
        // Add chat conversion history.
        let history = if let Some(memory) = &self.memory {
            let memory = memory.read().await;
            memory
                .messages()
                .iter()
                .map(|m| Message {
                    role: m.message_type.type_string(),
                    content: m.content.clone(),
                })
                .collect()
        } else {
            vec![]
        };
        self.chat(prompt, history).await
    }

    /// Processes a prompt using the agent.
    pub async fn chat(&self, prompt: &str, history: Vec<Message>) -> Result<String, TaskError> {
        let mut executor = Executor::new(
            self.model.clone(),
            self.knowledges.clone(),
            self.tools.clone(),
            self.memory.clone(),
            self.mcp_clients.clone(),
        );
        let mut req = Request::new(prompt.to_string(), self.preamble.clone());
        req.history = history;
        req.max_tokens = self.max_tokens;
        req.temperature = self.temperature;
        let tools = self.tools.read().await;
        req.tools = tools
            .iter()
            .map(|tool| tool.definition())
            .collect::<Vec<_>>();
        let mcp_clients = self.mcp_clients.read().await;
        for client in mcp_clients.iter() {
            for tool in client.tools.values() {
                req.tools.push(tool.clone());
            }
        }
        req.documents = stream::iter(self.store_indices.iter())
            .then(|(num_sample, storage)| async {
                Ok::<_, VectorStoreError>(
                    storage
                        .search(prompt, *num_sample, 0.5)
                        .await?
                        .into_iter()
                        .map(|(id, text, _)| Document {
                            id,
                            text,
                            additional_props: HashMap::new(),
                        })
                        .collect::<Vec<_>>(),
                )
            })
            .try_fold(vec![], |mut acc, docs| async {
                acc.extend(docs);
                Ok(acc)
            })
            .await
            .map_err(|err| TaskError::ExecutionError(err.to_string()))?;

        let response = executor
            .invoke(req)
            .await
            .map_err(|err| TaskError::ExecutionError(err.to_string()))?;

        Ok(response)
    }
}
