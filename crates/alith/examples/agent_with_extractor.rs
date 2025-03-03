use alith::{Extractor, LLM};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct Person {
    name: String,
    age: usize,
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let model = LLM::from_model_name("gpt-4")?;
    let extractor = Extractor::new::<Person>(model).await;
    let response: Person = extractor.extract("Alice is 18 years old").await?;
    println!("{:?}", response);
    Ok(())
}
