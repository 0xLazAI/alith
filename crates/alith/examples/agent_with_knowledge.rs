use alith::{Agent, Knowledge, PdfFileKnowledge, StringKnowledge, TextFileKnowledge, LLM};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let knowledges: Vec<Box<dyn Knowledge>> = vec![
        Box::new(StringKnowledge::new("Reference Joke 1")),
        Box::new(TextFileKnowledge::new("path/to/text.txt")),
        Box::new(PdfFileKnowledge::new("path/to/pdf.pdf")),
    ];
    let model = LLM::from_model_name("gpt-4o")?;
    let mut agent = Agent::new("simple agent", model, vec![]);
    agent.preamble =
        "You are a comedian here to entertain the user using humour and jokes.".to_string();
    agent.knowledges = Arc::new(knowledges);
    let response = agent.prompt("Entertain me!").await?;

    println!("{}", response);

    Ok(())
}
