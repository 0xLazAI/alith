use alith::{Agent, LLM};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let model = LLM::openai_compatible_model(
        std::env::var("LLM_API_KEY").unwrap_or_default().as_str(),
        "api.siliconflow.cn/v1",
        "deepseek-ai/DeepSeek-V3",
    )?;
    let agent = Agent::new("simple agent", model)
        .preamble("You are a comedian here to entertain the user using humour and jokes.");
    let response = agent.prompt("Entertain me!").await?;

    println!("{}", response);

    Ok(())
}
