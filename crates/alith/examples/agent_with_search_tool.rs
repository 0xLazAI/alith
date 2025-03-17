use alith::{Agent, LLM, SearchTool};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let model = LLM::from_model_name("gpt-4")?;
    let agent = Agent::new("simple agent", model)
        .preamble("You are a searcher. When I ask questions about Web3, you can search from the Internet and answer them. When you encounter other questions, you can directly answer them.")
        .tool(SearchTool::default()).await;
    let response = agent.prompt("What's BitCoin?").await?;

    println!("{}", response);

    Ok(())
}
