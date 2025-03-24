use crate::tool::ToolDefinition;
pub use mcp_client::Error;
pub use mcp_client::McpService;
pub use mcp_client::client::{ClientCapabilities, ClientInfo};
use mcp_client::client::{McpClient, McpClientTrait};
pub use mcp_client::transport::{SseTransport, StdioTransport, Transport};
use serde::Deserialize;
use std::collections::HashMap;
use std::ops::Deref;
use std::path::Path;
use std::time::Duration;
use thiserror::Error;
use tracing_subscriber::EnvFilter;

#[derive(Deserialize)]
pub struct MCPServerConfig {
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: HashMap<String, String>,
}

#[derive(Deserialize)]
pub struct MCPConfig {
    #[serde(rename = "mcpServers")]
    pub mcp_servers: HashMap<String, MCPServerConfig>,
}

pub struct MCPClient {
    pub client: Box<dyn McpClientTrait>,
    pub tools: HashMap<String, ToolDefinition>,
}

impl Deref for MCPClient {
    type Target = Box<dyn McpClientTrait>;

    fn deref(&self) -> &Self::Target {
        &self.client
    }
}

/// Init MCP logging.
#[inline]
pub fn init_logging() {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::from_default_env()
                .add_directive("mcp_client=debug".parse().unwrap())
                .add_directive("eventsource_client=info".parse().unwrap()),
        )
        .init();
}

/// Set up MCP clients config from the path, spawning each server,
/// and returning a HashMap<server_name -> Arc<Client>>.
/// spawn a single MCP process per server, share references.
pub async fn setup_mcp_clients<P: AsRef<Path>>(
    path: P,
) -> Result<HashMap<String, MCPClient>, MCPError> {
    let path = path.as_ref();
    let config_str = tokio::fs::read_to_string(&path).await?;
    let config: MCPConfig = serde_json::from_str(&config_str)?;

    let mut mcp_clients_map = HashMap::new();

    // For each server in the config, spawn an MCP client
    for (server_name, server_conf) in config.mcp_servers {
        let client = stdio_client(server_conf.command, server_conf.args, server_conf.env).await?;
        mcp_clients_map.insert(server_name, client);
    }

    Ok(mcp_clients_map)
}

#[derive(Error, Debug)]
pub enum MCPError {
    #[error("Failed to read config file: {0}")]
    ConfigReadError(#[from] std::io::Error),
    #[error("Failed to parse config file: {0}")]
    ConfigParseError(#[from] serde_json::Error),
    #[error("MCP error {0}")]
    MCPError(#[from] Error),
}

/// Create a sse mcp client.
pub async fn sse_client<S: AsRef<str>>(
    sse_url: S,
    env: HashMap<String, String>,
) -> Result<MCPClient, MCPError> {
    // Create the base transport
    let transport = SseTransport::new(sse_url.as_ref(), env);
    // Start transport
    let handle = transport.start().await.map_err(Error::Transport)?;
    // Create the service with timeout middleware
    let service = McpService::with_timeout(handle, Duration::from_secs(3));
    // Create client
    let mut client = McpClient::new(service);
    // Initialize
    client
        .initialize(
            ClientInfo {
                name: "alith-client".into(),
                version: "1.0.0".into(),
            },
            ClientCapabilities::default(),
        )
        .await?;
    // Sleep for 100ms to allow the server to start - surprisingly this is required!
    tokio::time::sleep(Duration::from_millis(100)).await;
    let tool_result = client.list_tools(None).await?;
    let mut tools = HashMap::new();
    for tool in tool_result.tools {
        tools.insert(
            tool.name.clone(),
            ToolDefinition {
                name: tool.name,
                description: tool.description,
                parameters: tool.input_schema,
            },
        );
    }
    Ok(MCPClient {
        client: Box::new(client),
        tools,
    })
}

pub async fn stdio_client<S: AsRef<str>>(
    command: S,
    args: Vec<S>,
    env: HashMap<String, String>,
) -> Result<MCPClient, MCPError> {
    // Create the base transport
    let transport = StdioTransport::new(
        command.as_ref().to_string(),
        args.iter().map(|s| s.as_ref().to_string()).collect(),
        env,
    );
    // Start transport
    let handle = transport.start().await.map_err(Error::Transport)?;
    // Create the service with timeout middleware
    let service = McpService::with_timeout(handle, Duration::from_secs(3));
    // Create client
    let mut client = McpClient::new(service);
    // Initialize
    client
        .initialize(
            ClientInfo {
                name: "alith-client".into(),
                version: "1.0.0".into(),
            },
            ClientCapabilities::default(),
        )
        .await?;
    // Sleep for 100ms to allow the server to start - surprisingly this is required!
    tokio::time::sleep(Duration::from_millis(100)).await;
    let tool_result = client.list_tools(None).await?;
    let mut tools = HashMap::new();
    for tool in tool_result.tools {
        tools.insert(
            tool.name.clone(),
            ToolDefinition {
                name: tool.name,
                description: tool.description,
                parameters: tool.input_schema,
            },
        );
    }
    Ok(MCPClient {
        client: Box::new(client),
        tools,
    })
}
