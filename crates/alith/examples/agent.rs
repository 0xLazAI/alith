use alith::{Agent, LLM};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let model = LLM::from_model_name("gpt-4")?;
    let mut agent = Agent::new("simple agent", model, vec![]);
    agent.preamble =
        "You are a comedian here to entertain the user using humour and jokes.".to_string();
    let response = agent.prompt("Entertain me!").await?;

    println!("{}", response);

    Ok(())
}
