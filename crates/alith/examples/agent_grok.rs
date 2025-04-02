use alith::{Agent, Chat, LLM};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let model = LLM::openai_compatible_model(
        std::env::var("GROK_API_KEY").unwrap().as_str(),
        "api.grok.ai/v1",
        "grok-3",
    )?;
    let agent = Agent::new("simple agent", model)
        .preamble("You are a comedian here to entertain the user using humour and jokes.");
    let response = agent.prompt("Entertain me!").await?;

    println!("{}", response);

    Ok(())
}
