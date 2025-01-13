use alith::{Agent, StructureTool, Tool, ToolError, LLM};
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
    let tools: [Box<dyn Tool>; 2] = [Box::new(Adder), Box::new(Subtract)];
    let model = LLM::from_model_name("gpt-4o")?;
    let mut agent = Agent::new("simple agent", model, tools);
    agent.preamble =
        "You are a calculator here to help the user perform arithmetic operations. Use the tools provided to answer the user's question.".to_string();
    let response = agent.prompt("Calculate 10 - 3").await?;

    println!("{}", response);

    Ok(())
}
