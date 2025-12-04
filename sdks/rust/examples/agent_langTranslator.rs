use alith::{Agent, Chat, LLM};

#[tokio::main]
async fn main() {
    let model = LLM::openai_compatible_model(
        "Your-api-key",
        "https://api.groq.com/openai/v1",
        "llama-3.1-8b-instant",
    )
    .expect("Failed to create model");

    let agent = Agent::new("french translator agent", model).preamble(
        "You are a professional English-to-French translator. \
         You accurately translate English text into natural, fluent, and grammatically correct French. \
         Preserve the tone, emotion, and context of the original text. \
         Do not explain or add extra commentary—return only the translated text.",
    );

    // Translation request parameters
    let text = "Artificial intelligence is transforming the way we live and work.";
    let tone = "formal"; // can be "formal", "casual", "neutral", etc.

    let prompt = format!(
        r#"
    Translate the following English text into French.
    Maintain a {} tone and ensure the translation sounds natural to native French speakers.

    Text to translate:
    "{}"
    "#,
        tone, text
    );

    let response = agent.prompt(&prompt).await.expect("Failed to get response");

    println!("🇫🇷 Translated Text:\n{}", response);
}
