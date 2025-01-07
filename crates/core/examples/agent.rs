use alith_core::{agent::Agent, llm::LLM};
use std::{env, sync::Arc};
use tokio::sync::RwLock;
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let model = LLM::from_model_name("openai")?
        .with_api_key(&env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set"));
    let mut agent = Agent::new(
        Arc::new(RwLock::new(model)),
        Arc::new(Vec::new()),
        Uuid::from_u128(123),
        "simple agent".to_string(),
    );
    agent.preamble =
        "You are a comedian here to entertain the user using humour and jokes.".to_string();
    let response = agent.prompt("Entertain me!").await?;

    println!("{}", response);

    Ok(())
}
