use alith::{
    Agent, Chat, HtmlKnowledge, Knowledge, LLM, PdfFileKnowledge, StringKnowledge,
    TextFileKnowledge,
};
use std::io::Cursor;
use std::sync::Arc;
use url::Url;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let url = "https://en.m.wikivoyage.org/wiki/Seoul";
    let html = reqwest::get(url).await.unwrap().text().await.unwrap();

    let knowledges: Vec<Box<dyn Knowledge>> = vec![
        Box::new(StringKnowledge::new("Reference Joke 1")),
        Box::new(TextFileKnowledge::new("path/to/text.txt")),
        Box::new(PdfFileKnowledge::new("path/to/pdf.pdf")),
        Box::new(HtmlKnowledge::new(
            Cursor::new(html),
            Url::parse(url).unwrap(),
            false,
        )),
    ];
    let model = LLM::from_model_name("gpt-4")?;
    let mut agent = Agent::new("simple agent", model)
        .preamble("You are a comedian here to entertain the user using humour and jokes.");
    agent.knowledges = Arc::new(knowledges);
    let response = agent.prompt("Entertain me!").await?;

    println!("{}", response);

    Ok(())
}
