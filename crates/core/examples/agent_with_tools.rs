use alith_core::{
    agent::Agent,
    llm::LLM,
    tool::{StructureTool, ToolError},
};
use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::{env, sync::Arc};
use tokio::sync::RwLock;
use uuid::Uuid;

#[derive(JsonSchema, Serialize, Deserialize)]
pub struct Input {
    pub x: usize,
    pub y: usize,
}

pub struct Adder;
#[async_trait]
impl StructureTool for Adder {
    type Input = Input;
    type Output = String;

    fn name(&self) -> &str {
        "adder"
    }

    fn description(&self) -> &str {
        "Add x and y together"
    }

    async fn run_with_args(&self, input: Self::Input) -> Result<Self::Output, ToolError> {
        let result = input.x + input.y;
        Ok(result.to_string())
    }
}

pub struct Subtract;
#[async_trait]
impl StructureTool for Subtract {
    type Input = Input;
    type Output = String;

    fn name(&self) -> &str {
        "subtract"
    }

    fn description(&self) -> &str {
        "Subtract y from x (i.e.: x - y)"
    }

    async fn run_with_args(&self, input: Self::Input) -> Result<Self::Output, ToolError> {
        let result = input.x - input.y;
        Ok(result.to_string())
    }
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let model = LLM::from_model_name("openai")?
        .with_api_key(&env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set"));
    let mut agent = Agent::new(
        Arc::new(RwLock::new(model)),
        Arc::new(vec![Box::new(Adder), Box::new(Subtract)]),
        Uuid::from_u128(123),
        "simple agent".to_string(),
    );
    agent.preamble =
        "You are a calculator here to help the user perform arithmetic operations. Use the tools provided to answer the user's question.".to_string();
    let response = agent.prompt("Calculate 10 - 3").await?;

    println!("{}", response);

    Ok(())
}
