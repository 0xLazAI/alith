use alith::{Agent, Chat, inference::MistralRsEngine};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let model = MistralRsEngine::new("/root/models/qwen2.5-1.5b-instruct-q5_k_m.gguf").await?;
    let agent = Agent::new("simple agent", model);
    println!("{}", agent.prompt("Calculate 10 - 3").await?);
    Ok(())
}
