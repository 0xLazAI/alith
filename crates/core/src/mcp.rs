use anyhow::Result;
pub use mcp_client::McpService;
pub use mcp_client::client::{ClientCapabilities, ClientInfo, McpClient, McpClientTrait};
pub use mcp_client::transport::{SseTransport, StdioTransport, Transport};
use std::collections::HashMap;
use std::time::Duration;
use tracing_subscriber::EnvFilter;

pub async fn sse_client<S: AsRef<str>>(
    sse_url: S,
    env: HashMap<String, String>,
) -> Result<impl McpClientTrait> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::from_default_env()
                .add_directive("mcp_client=debug".parse().unwrap())
                .add_directive("eventsource_client=info".parse().unwrap()),
        )
        .init();
    // Create the base transport
    let transport = SseTransport::new(sse_url.as_ref(), env);
    // Start transport
    let handle = transport.start().await?;
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
    Ok(client)
}

pub async fn stdio_client<S: AsRef<str>>(
    command: S,
    args: Vec<S>,
    env: HashMap<String, String>,
) -> Result<impl McpClientTrait> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::from_default_env()
                .add_directive("mcp_client=debug".parse().unwrap())
                .add_directive("eventsource_client=info".parse().unwrap()),
        )
        .init();
    // Create the base transport
    let transport = StdioTransport::new(
        command.as_ref().to_string(),
        args.iter().map(|s| s.as_ref().to_string()).collect(),
        env,
    );
    // Start transport
    let handle = transport.start().await?;
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
    Ok(client)
}
