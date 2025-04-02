use alith::{Agent, Chat, LLM};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let model = LLM::from_model_name("gpt-4")?;
    let agent = Agent::new("simple agent", model)
        .preamble("You are a comedian here to entertain the user using humour and jokes.");

    let response = agent.prompt("Entertain me!").await?;

    println!("{}", response);

    Ok(())
}
