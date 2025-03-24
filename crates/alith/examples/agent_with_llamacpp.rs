#[cfg(not(target_os = "windows"))]
use alith::{Agent, inference::LlamaEngine};

#[cfg(not(target_os = "windows"))]
#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let model = LlamaEngine::new("/root/models/qwen2.5-1.5b-instruct-q5_k_m.gguf").await?;
    let agent = Agent::new("simple agent", model);
    println!("{}", agent.prompt("Calculate 10 - 3").await?);
    Ok(())
}

#[cfg(target_os = "windows")]
fn main() {}
