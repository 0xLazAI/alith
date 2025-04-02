use alith::{
    Agent, Chat,
    inference::engines::ort::{GraphOptimizationLevel, ort_init, present::GPT2},
};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    ort_init()?;
    let model = GPT2::new(
        "https://cdn.pyke.io/0/pyke:ort-rs/example-models@0.0.0/gpt2.onnx",
        "tokenizer.json",
        GraphOptimizationLevel::Level1,
        1,
    )?;
    let agent = Agent::new("simple agent", model);
    println!("{}", agent.prompt("Calculate 10 - 3").await?);
    Ok(())
}
