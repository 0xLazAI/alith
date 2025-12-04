use alith::{Agent, Chat, LLM};

#[tokio::main]
async fn main() {
    let model = LLM::openai_compatible_model(
        "Your-api-key",
        "https://api.groq.com/openai/v1",
        "llama-3.1-8b-instant",
    )
    .expect("Failed to create model");

    let agent = Agent::new("github debug agent", model)
        .preamble("You are an expert in version control systems like Git and GitHub. Analyze error messages and provide accurate solutions step by step.");

    let error_message =
        "error: Your local changes to the following files would be overwritten by merge: ";

    let suggestion = agent
        .prompt(error_message)
        .await
        .expect("Failed to get response");

    println!("Suggestion: {}", suggestion);
}
