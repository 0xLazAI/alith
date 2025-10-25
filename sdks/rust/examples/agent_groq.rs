use alith::{Agent, Chat, LLM};

#[tokio::main]
async fn main() {
    let model = LLM::openai_compatible_model(
        "Your-api-key",
        "api.groq.com/openai/v1",
        "llama-3.1-8b-instant",
    )
    .expect("Failed to create model");

    let agent = Agent::new("calculator agent", model)
        .preamble("You are a calculator here to help the user perform arithmetic operations. Use the tools provided to answer the user question.");

    let response = agent
        .prompt("Calculate 10 - 3")
        .await
        .expect("Failed to get response");

    println!("{}", response);
}
