use alith::{Agent, SearchTool, Tool, LLM};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let tools: [Box<dyn Tool>; 1] = [Box::new(SearchTool::default())];
    let model = LLM::from_model_name("gpt-4")?;
    let mut agent = Agent::new("simple agent", model, tools);
    agent.preamble =
        "You are a searcher. When I ask questions about Web3, you can search from the Internet and answer them. When you encounter other questions, you can directly answer them.".to_string();
    let response = agent.prompt("What's BitCoin?").await?;

    println!("{}", response);

    Ok(())
}
