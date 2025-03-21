use alith::inference::MistralRsEngine;
use alith::{Agent, StructureTool, ToolError};
use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(JsonSchema, Serialize, Deserialize)]
pub struct Input {
    pub x: usize,
    pub y: usize,
}

pub struct Adder;
#[async_trait]
impl StructureTool for Adder {
    type Input = Input;
    type Output = usize;

    fn name(&self) -> &str {
        "adder"
    }

    fn description(&self) -> &str {
        "Add x and y together"
    }

    async fn run_with_args(&self, input: Self::Input) -> Result<Self::Output, ToolError> {
        let result = input.x + input.y;
        Ok(result)
    }
}

pub struct Subtract;
#[async_trait]
impl StructureTool for Subtract {
    type Input = Input;
    type Output = usize;

    fn name(&self) -> &str {
        "subtract"
    }

    fn description(&self) -> &str {
        "Subtract y from x (i.e.: x - y)"
    }

    async fn run_with_args(&self, input: Self::Input) -> Result<Self::Output, ToolError> {
        let result = input.x - input.y;
        Ok(result)
    }
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let model = MistralRsEngine::new("/root/models/qwen2.5-1.5b-instruct-q5_k_m.gguf").await?;
    let agent = Agent::new("simple agent", model)
        .preamble("You are a calculator here to help the user perform arithmetic operations. Use the tools provided to answer the user's question.");
    let response = agent.prompt("Calculate 10 - 3").await?;

    println!("{}", response);

    Ok(())
}
