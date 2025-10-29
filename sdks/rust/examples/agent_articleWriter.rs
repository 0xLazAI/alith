use alith::{Agent, Chat, LLM};

#[tokio::main]
async fn main() {
    let model = LLM::openai_compatible_model(
        "Your-api-key",
        "https://api.groq.com/openai/v1",
        "llama-3.1-8b-instant",
    )
    .expect("Failed to create model");

    let agent = Agent::new("article writer agent", model)
        .preamble("You are an expert article writer who produces well-structured, engaging, and SEO-friendly articles. Always provide clear section titles, concise paragraphs, and maintain a professional tone.");

    // Article request parameters
    let topic = "The Future of Artificial Intelligence";
    let target_audience = "tech enthusiasts";
    let tone = "informative and futuristic";
    let word_count = 600;

    let prompt = format!(
        r#"
    Topic: {}
    Target Audience: {}
    Tone: {}
    Desired Word Count: {}

    Write a complete article following these rules:
    1. Include a title and 4–6 clear sections with headings.
    2. Use engaging and professional language.
    3. Maintain the requested tone.
    4. Keep total length around {} words.
    5. End with a short conclusion or takeaway.
    "#,
        topic, target_audience, tone, word_count, word_count
    );

    let response = agent.prompt(&prompt).await.expect("Failed to get response");

    println!("{}", response);
}
