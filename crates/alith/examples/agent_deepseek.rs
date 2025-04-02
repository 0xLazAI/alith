use alith::{Agent, Chat, LLM};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let model = LLM::openai_compatible_model(
        "<Your API Key>", // Replace with your api key or read it from env.
        "api.deepseek.com",
        "deepseek-chat", // or `deepseek-reasoner` for DeepSeek R1
    )?;
    let agent = Agent::new("simple agent", model)
        .preamble("You are a comedian here to entertain the user using humour and jokes.");
    let response = agent.prompt("Entertain me!").await?;

    println!("{}", response);

    Ok(())
}
