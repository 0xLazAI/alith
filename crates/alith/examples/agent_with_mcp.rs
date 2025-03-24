use alith::{Agent, LLM};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let agent = Agent::new("", LLM::from_model_name("gpt-4")?)
        .mcp_config_path("examples/servers_config.json")
        .await?;
    println!("{}", agent.prompt("Read the content of Cargo.toml").await?);
    Ok(())
}
